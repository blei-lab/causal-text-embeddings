from semi_parametric_estimation.att import att_estimates
from reddit.data_cleaning.reddit_posts import load_reddit_processed
from supervised_lda.helpers import filter_document_terms, make_index_mapping, assign_split, tokenize_documents
import numpy as np
import pandas as pd
import os
from supervised_lda.supervised_topic_model import SupervisedTopicModel
from sklearn.linear_model import LogisticRegression, Ridge
from supervised_lda import run_supervised_tm
from sklearn.metrics import mean_squared_error as mse
import argparse
import sys
from scipy.special import logit
from scipy import sparse

def load_term_counts(reddit, path='../dat/reddit/', force_redo=False):
	count_filename = path  + 'term_counts'
	vocab_filename = path + 'vocab'

	if os.path.exists(count_filename + '.npz') and not force_redo:
		return sparse.load_npz(count_filename + '.npz').toarray(), np.load(vocab_filename + '.npy')

	post_docs = reddit['post_text'].values
	counts, vocab, _ = tokenize_documents(post_docs)    
	sparse.save_npz(count_filename, counts)
	np.save(vocab_filename, vocab)
	return counts.toarray(), np.array(vocab)

def load_simulated_data():
	sim_df = pd.read_csv(simulation_file, delimiter='\t')
	sim_df = sim_df.rename(columns={'index':'post_index'})
	return sim_df

def drop_empty_posts(counts):
	doc_terms = counts.sum(axis=1)
	return doc_terms >= 5

def fit_model(doc_embeddings, labels, is_binary=False):
	if is_binary:
		model = LogisticRegression(solver='liblinear')
	else:
		model = Ridge()
	model.fit(doc_embeddings, labels)
	return model

def main():
	if dat_dir:
		reddit = load_reddit_processed(path=dat_dir)
	else:
		reddit = load_reddit_processed()

	if subs:
		reddit = reddit[reddit.subreddit.isin(subs)]
	reddit = reddit.dropna(subset=['post_text'])

	
	index_mapping = make_index_mapping(reddit, on='orig_index')
	if not dat_dir:
		counts, vocab = load_term_counts(reddit)
	else:
		counts, vocab = load_term_counts(reddit, path=dat_dir)

	sim_df = load_simulated_data()

	train_df = sim_df[sim_df.split != split]
	predict_df = sim_df[sim_df.split == split]

	tr_treatment_labels = train_df.treatment.values
	tr_outcomes = train_df.outcome.values
	predict_treatment = predict_df.treatment.values
	predict_outcomes = predict_df.outcome.values

	tr_counts = filter_document_terms(train_df, counts, index_mapping)
	predict_counts = filter_document_terms(predict_df, counts, index_mapping)
	tr_valid = drop_empty_posts(tr_counts)
	pred_valid = drop_empty_posts(predict_counts)
	tr_counts = tr_counts[tr_valid, :]
	predict_counts = predict_counts[pred_valid, :]

	tr_treatment_labels = tr_treatment_labels[tr_valid]
	tr_outcomes = tr_outcomes[tr_valid]
	predict_treatment = predict_treatment[pred_valid]
	predict_outcomes = predict_outcomes[pred_valid]

	num_documents = tr_counts.shape[0]
	vocab_size = tr_counts.shape[1]
	model = SupervisedTopicModel(num_topics, vocab_size, num_documents, outcome_linear_map=linear_outcome_model)

	run_supervised_tm.train(model, tr_counts, tr_treatment_labels, tr_outcomes, num_epochs=num_iters, use_recon_loss=use_recon_loss, use_sup_loss=use_supervised_loss)

	if use_supervised_loss:
		propensity_score, expected_outcome_treat, expected_outcome_no_treat = run_supervised_tm.predict(model, predict_counts)
	else:
		tr_doc_embeddings = run_supervised_tm.get_representation(model, tr_counts)
		treated = tr_treatment_labels == 1
		out_treat = tr_outcomes[treated]
		out_no_treat = tr_outcomes[~treated]
		q0_embeddings = tr_doc_embeddings[~treated,:]
		q1_embeddings = tr_doc_embeddings[treated,:]
		q0_model = fit_model(q0_embeddings, out_no_treat)
		q1_model = fit_model(q1_embeddings, out_treat)
		g_model = fit_model(tr_doc_embeddings, tr_treatment_labels, is_binary=True)

		pred_doc_embeddings = run_supervised_tm.get_representation(model, predict_counts)
		propensity_score = g_model.predict_proba(pred_doc_embeddings)[:,1]
		expected_outcome_no_treat = q0_model.predict(pred_doc_embeddings)
		expected_outcome_treat = q1_model.predict(pred_doc_embeddings)
		
	out = os.path.join(outdir, str(split))
	os.makedirs(out, exist_ok=True)
	outfile = os.path.join(out, 'predictions')
	np.savez_compressed(outfile, g=propensity_score, q0=expected_outcome_no_treat, q1=expected_outcome_treat, t=predict_treatment, y=predict_outcomes)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dat-dir", action="store", default=None)
	parser.add_argument("--outdir", action="store", default='../out/')
	parser.add_argument("--sim-dir", action="store", default='../dat/sim/reddit_subreddit_based/')
	parser.add_argument("--subs", action="store", default='13,6,8')
	parser.add_argument("--mode", action="store", default="simple")
	parser.add_argument("--params", action="store", default="1.0,1.0,1.0")
	parser.add_argument("--verbose", action='store_true')
	parser.add_argument("--num-topics", action='store', default=100)
	parser.add_argument("--split", action='store', default=0)
	parser.add_argument("--num-iters", action="store", default=4000)
	# parser.add_argument("--num_splits", action='store', default=10)
	parser.add_argument("--linear-outcome-model", action='store', default="t")
	parser.add_argument("--use-recon-loss", action='store', default="t")
	parser.add_argument("--use-supervised-loss", action='store', default="t")
	args = parser.parse_args()

	sim_dir = args.sim_dir
	dat_dir = args.dat_dir
	outdir = args.outdir
	subs = None
	if args.subs != '':
		subs = [int(s) for s in args.subs.split(',')]
	verbose = args.verbose
	params = args.params.split(',')
	sim_setting = 'beta0' + params[0] + '.beta1' + params[1] + '.gamma' + params[2]
	subs_string = ', '.join(args.subs.split(','))
	mode = args.mode
	simulation_file = sim_dir + 'subreddits['+ subs_string + ']/mode' + mode + '/' + sim_setting + ".tsv"
	num_iters = int(args.num_iters)
	num_topics = int(args.num_topics)
	split = int(args.split)
	# num_splits = args.num_splits
	linear_outcome_model = True if args.linear_outcome_model == "t" else False 
	use_supervised_loss = True if args.use_supervised_loss == "t" else False
	use_recon_loss = True if args.use_recon_loss == "t" else False

	main()