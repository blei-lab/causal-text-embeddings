from semi_parametric_estimation.att import att_estimates
from supervised_lda.helpers import filter_document_terms, make_index_mapping, assign_split, tokenize_documents
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error as mse
import argparse
import sys
from supervised_lda.supervised_topic_model import SupervisedTopicModel
from supervised_lda import run_supervised_tm
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.special import logit

def load_peerread(path='../dat/PeerRead/'):
	return pd.read_csv(path + 'proc_abstracts.csv')

def load_term_counts(df, path='../dat/PeerRead/', force_redo=False, text_col='abstract_text'):
	count_filename = path  + 'term_counts'
	vocab_filename = path + 'vocab'

	if os.path.exists(count_filename + '.npz') and not force_redo:
		return sparse.load_npz(count_filename + '.npz').toarray(), np.load(vocab_filename + '.npy')

	post_docs = df[text_col].values
	counts, vocab, _ = tokenize_documents(post_docs)    
	sparse.save_npz(count_filename, counts)
	np.save(vocab_filename, vocab)
	return counts.toarray(), np.array(vocab)

def compute_ground_truth_treatment_effect(df):
	y1 = df['y1']
	y0 = df['y0']
	return y1.mean() - y0.mean()

def load_simulated_data():
	sim_df = pd.read_csv(simulation_file, delimiter='\t')
	return sim_df

def fit_model(doc_embeddings, labels, is_binary=False):
	if is_binary:
		model = LogisticRegression(solver='liblinear')
	else:
		model = Ridge()
	model.fit(doc_embeddings, labels)
	return model

def main():
	if dat_dir:
		peerread = load_peerread(path=dat_dir)
		counts,vocab = load_term_counts(peerread,path=dat_dir)
	else:
		peerread = load_peerread()
		counts,vocab = load_term_counts(peerread)

	indices = peerread['paper_id'].values
	index_mapping = make_index_mapping(peerread, on='index')

	sim_df = load_simulated_data()

	train_df = sim_df[sim_df.split != split]
	predict_df = sim_df[sim_df.split == split]
	tr_treatment_labels = train_df.treatment.values
	tr_outcomes = train_df.outcome.values
	predict_treatment = predict_df.treatment.values
	predict_outcomes = predict_df.outcome.values

	tr_counts = filter_document_terms(train_df, counts, index_mapping, on='id')
	predict_counts = filter_document_terms(predict_df, counts, index_mapping, on='id')

	num_documents = tr_counts.shape[0]
	vocab_size = tr_counts.shape[1]
	model = SupervisedTopicModel(num_topics, vocab_size, num_documents, outcome_linear_map=linear_outcome_model)

	run_supervised_tm.train(model, tr_counts, tr_treatment_labels, tr_outcomes, dtype='binary', 
		num_epochs=num_iters, use_recon_loss=use_recon_loss, use_sup_loss=use_supervised_loss)

	if use_supervised_loss:
		propensity_score, expected_outcome_treat, expected_outcome_no_treat = run_supervised_tm.predict(model, predict_counts, dtype='binary')
	else:
		tr_doc_embeddings = run_supervised_tm.get_representation(model, tr_counts)
		treated = tr_treatment_labels == 1
		out_treat = tr_outcomes[treated]
		out_no_treat = tr_outcomes[~treated]
		q0_embeddings = tr_doc_embeddings[~treated,:]
		q1_embeddings = tr_doc_embeddings[treated,:]
		q0_model = fit_model(q0_embeddings, out_no_treat, is_binary=True)
		q1_model = fit_model(q1_embeddings, out_treat, is_binary=True)
		g_model = fit_model(tr_doc_embeddings, tr_treatment_labels, is_binary=True)

		pred_doc_embeddings = run_supervised_tm.get_representation(model, predict_counts)
		propensity_score = g_model.predict_proba(pred_doc_embeddings)[:,1]
		expected_outcome_no_treat = q0_model.predict_proba(pred_doc_embeddings)[:,1]
		expected_outcome_treat = q1_model.predict_proba(pred_doc_embeddings)[:,1]

	out = os.path.join(outdir, str(split))
	os.makedirs(out, exist_ok=True)
	outfile = os.path.join(out, 'predictions')
	np.savez_compressed(outfile, g=propensity_score, q0=expected_outcome_no_treat, q1=expected_outcome_treat, t=predict_treatment, y=predict_outcomes)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dat-dir", action="store", default=None)
	parser.add_argument("--outdir", action="store", default='../out/')
	parser.add_argument("--sim-dir", action="store", default='../dat/sim/peerread_buzzytitle_based/')
	parser.add_argument("--mode", action="store", default="simple")
	parser.add_argument("--params", action="store", default="1.0")
	parser.add_argument("--verbose", action='store_true')
	parser.add_argument("--split", action='store', default=0)
	parser.add_argument("--num-iters", action="store", default=3000)
	parser.add_argument("--num-topics", action='store', default=100)
	parser.add_argument("--linear-outcome-model", action='store', default="t")
	parser.add_argument("--use-recon-loss", action='store', default="t")
	parser.add_argument("--use-supervised-loss", action='store', default="t")
	args = parser.parse_args()

	sim_dir = args.sim_dir
	outdir = args.outdir
	dat_dir = args.dat_dir
	verbose = args.verbose
	params = args.params
	sim_setting = 'beta00.25' + '.beta1' + params + '.gamma0.0'
	mode = args.mode
	simulation_file = sim_dir + '/mode' + mode + '/' + sim_setting + ".tsv"
	num_topics = args.num_topics
	split = int(args.split)
	linear_outcome_model = True if args.linear_outcome_model == "t" else False 
	use_supervised_loss = True if args.use_supervised_loss == "t" else False
	use_recon_loss = True if args.use_recon_loss == "t" else False
	num_iters = int(args.num_iters)
	print(use_supervised_loss, use_recon_loss, linear_outcome_model)

	main()