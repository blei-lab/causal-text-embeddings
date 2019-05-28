from semi_parametric_estimation.att import att_estimates
from reddit.data_cleaning.reddit_posts import load_reddit_processed
from .helpers import filter_document_embeddings, make_index_mapping, assign_split
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error as mse
import argparse
import sys
from scipy.special import logit

def get_log_outcomes(outcomes):
	#relu
	outcomes = np.array([max(0.0, out) + 1.0  for out in outcomes])
	return np.log(outcomes)

def predict_expected_outcomes(model, doc_embeddings):
	features = logit(doc_embeddings)
	return model.predict(features)

def fit_conditional_expected_outcomes(outcomes, doc_embeddings):
	model = LinearRegression()
	features = logit(doc_embeddings)
	model.fit(features, outcomes)
	predict = model.predict(features)
	if verbose:
		print("Training MSE:", mse(outcomes, predict))
	return model

def predict_treatment_probability(labels, doc_embeddings):
	model = LogisticRegression(solver='liblinear')
	features = logit(doc_embeddings)
	model.fit(features, labels)
	if verbose:
		print("Training accuracy:", model.score(features, labels))
	treatment_probability = model.predict_proba(features)[:,1]
	return treatment_probability

def load_simulated_data():
	sim_df = pd.read_csv(simulation_file, delimiter='\t')
	sim_df = sim_df.rename(columns={'index':'post_index'})
	return sim_df

def load_document_proportions(path='../dat/reddit/'):
	return np.load(path + 'document_proportions.npy')

def main():
	reddit = load_reddit_processed()
	if subs:
		reddit = reddit[reddit.subreddit.isin(subs)]

	index_mapping = make_index_mapping(reddit, on='orig_index')
	if not dat_dir:
		doc_embeddings = load_document_proportions()
	else:
		doc_embeddings = load_document_proportions(path=dat_dir)

	sim_df = load_simulated_data()
	num_reps = 10
	mean_estimates = {}

	for rep in range(num_reps):
		bootstrap_sim_df = assign_split(sim_df, num_splits=2)
		bootstrap_sim_df = bootstrap_sim_df[bootstrap_sim_df.split==0]
		treatment_labels = bootstrap_sim_df.treatment.values
		filtered_doc_embeddings = filter_document_embeddings(bootstrap_sim_df, doc_embeddings, index_mapping)
		treatment_probability = predict_treatment_probability(treatment_labels, filtered_doc_embeddings)

		treated_sim = bootstrap_sim_df[bootstrap_sim_df.treatment==1]
		untreated_sim = bootstrap_sim_df[bootstrap_sim_df.treatment==0]
		
		all_outcomes = bootstrap_sim_df.outcome.values
		outcomes_st_treated = treated_sim.outcome.values
		outcomes_st_not_treated = untreated_sim.outcome.values
		
		doc_embed_st_treated = filter_document_embeddings(treated_sim, doc_embeddings, index_mapping)
		doc_embed_st_not_treated = filter_document_embeddings(untreated_sim, doc_embeddings, index_mapping)

		model_outcome_st_treated = fit_conditional_expected_outcomes(outcomes_st_treated, doc_embed_st_treated)
		model_outcome_st_not_treated = fit_conditional_expected_outcomes(outcomes_st_not_treated, doc_embed_st_not_treated)

		expected_outcome_st_treated = predict_expected_outcomes(model_outcome_st_treated, filtered_doc_embeddings)
		expected_outcome_st_not_treated = predict_expected_outcomes(model_outcome_st_not_treated, filtered_doc_embeddings)

		estimates = att_estimates(expected_outcome_st_not_treated, expected_outcome_st_treated, 
			treatment_probability, treatment_labels, all_outcomes, truncate_level=0.03, prob_t=treatment_labels.mean())

		for est, ate in estimates.items():
			if est in mean_estimates:
				mean_estimates[est].append(ate)
			else:
				mean_estimates[est] = [ate]
	if verbose:
		for est, ates in mean_estimates.items():
			print(est, np.mean(ates), np.std(ates))
	else:
		config = ';'.join([str(mode)] + params)
		log_file = os.path.join(sim_dir, 'two-stage-lda-estimates.out')
		with open(log_file, 'a') as h:
			h.write(config + '\n')
			for est, ates in mean_estimates.items():
				h.write(est + ',' +  str(np.mean(ates)) + ',' + str(np.std(ates)) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dat-dir", action="store", default=None)
	parser.add_argument("--sim-dir", action="store", default='../dat/sim/reddit_subreddit_based/')
	parser.add_argument("--subs", action="store", default='13,6,8')
	parser.add_argument("--mode", action="store", default="simple")
	parser.add_argument("--params", action="store", default="1.0,1.0,1.0")
	parser.add_argument("--verbose", action='store_true')
	args = parser.parse_args()

	sim_dir = args.sim_dir
	dat_dir = args.dat_dir
	subs = None
	if args.subs != '':
		subs = [int(s) for s in args.subs.split(',')]
	verbose = args.verbose
	params = args.params.split(',')
	sim_setting = 'beta0' + params[0] + '.beta1' + params[1] + '.gamma' + params[2]
	subs_string = ', '.join(args.subs.split(','))
	mode = args.mode
	simulation_file = sim_dir + 'subreddits['+ subs_string + ']/mode' + mode + '/' + sim_setting + ".tsv"

	main()