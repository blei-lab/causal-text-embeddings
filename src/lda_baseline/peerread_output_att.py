from semi_parametric_estimation.ate import ate_estimates
from .peerread_fit_topics import load_peerread
from .helpers import filter_document_embeddings, make_index_mapping, assign_split
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error as mse
import argparse
import sys
from scipy.special import logit

def compute_ground_truth_treatment_effect(df):
	y1 = df['y1']
	y0 = df['y0']
	return y1.mean() - y0.mean()

def get_log_outcomes(outcomes):
	#relu
	outcomes = np.array([max(0.0, out) + 1.0  for out in outcomes])
	return np.log(outcomes)

def predict_expected_outcomes(model, doc_embeddings):
	features = logit(doc_embeddings)
	return model.predict_proba(features)[:,1]

def fit_conditional_expected_outcomes(outcomes, doc_embeddings):
	model = LogisticRegression(solver='liblinear')
	features = logit(doc_embeddings)
	model.fit(features, outcomes)
	if verbose:
		print("Training accuracy:", model.score(features, outcomes))
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
	return sim_df

def load_document_proportions(path='../dat/PeerRead/'):
	return np.load(path + 'document_proportions.npy')

def main():
	peerread = load_peerread()
	indices = peerread['paper_id'].values
	index_mapping = make_index_mapping(peerread, on='index')

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
		filtered_doc_embeddings = filter_document_embeddings(bootstrap_sim_df, doc_embeddings, index_mapping, on='id')
		treatment_probability = predict_treatment_probability(treatment_labels, filtered_doc_embeddings)

		treated_sim = bootstrap_sim_df[bootstrap_sim_df.treatment==1]
		untreated_sim = bootstrap_sim_df[bootstrap_sim_df.treatment==0]
		
		all_outcomes = bootstrap_sim_df.outcome.values
		outcomes_st_treated = treated_sim.outcome.values
		outcomes_st_not_treated = untreated_sim.outcome.values
		
		doc_embed_st_treated = filter_document_embeddings(treated_sim, doc_embeddings, index_mapping, on='id')
		doc_embed_st_not_treated = filter_document_embeddings(untreated_sim, doc_embeddings, index_mapping, on='id')

		model_outcome_st_treated = fit_conditional_expected_outcomes(outcomes_st_treated, doc_embed_st_treated)
		model_outcome_st_not_treated = fit_conditional_expected_outcomes(outcomes_st_not_treated, doc_embed_st_not_treated)

		expected_outcome_st_treated = predict_expected_outcomes(model_outcome_st_treated, filtered_doc_embeddings)
		expected_outcome_st_not_treated = predict_expected_outcomes(model_outcome_st_not_treated, filtered_doc_embeddings)

		estimates = ate_estimates(expected_outcome_st_not_treated, expected_outcome_st_treated, 
			treatment_probability, treatment_labels, all_outcomes, truncate_level=0.03)

		for est, ate in estimates.items():
			if est in mean_estimates:
				mean_estimates[est].append(ate)
			else:
				mean_estimates[est] = [ate]

	ground_truth_ate = compute_ground_truth_treatment_effect(sim_df)
	mean_estimates.update({'ground_truth_ate':ground_truth_ate})
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
	parser.add_argument("--sim-dir", action="store", default='../dat/sim/peerread_buzzytitle_based/')
	parser.add_argument("--mode", action="store", default="simple")
	parser.add_argument("--params", action="store", default="1.0")
	parser.add_argument("--verbose", action='store_true')
	args = parser.parse_args()

	sim_dir = args.sim_dir
	dat_dir = args.dat_dir
	verbose = args.verbose
	params = args.params
	sim_setting = 'beta00.25' + '.beta1' + params + '.gamma0.0'
	mode = args.mode
	simulation_file = sim_dir + '/mode' + mode + '/' + sim_setting + ".tsv"

	main()