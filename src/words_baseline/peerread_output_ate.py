from semi_parametric_estimation.ate import psi_q_only,psi_tmle_cont_outcome
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error as mse
import argparse
import sys
from scipy.special import logit
from scipy.sparse import load_npz

def compute_ground_truth_treatment_effect(df):
	y1 = df['y1']
	y0 = df['y0']
	return y1.mean() - y0.mean()

def get_log_outcomes(outcomes):
	#relu
	outcomes = np.array([max(0.0, out) + 1.0  for out in outcomes])
	return np.log(outcomes)

def predict_expected_outcomes(model, features):
	return model.predict_proba(features)[:,1]

def fit_conditional_expected_outcomes(outcomes, features):
	model = LogisticRegression(solver='liblinear')
	model.fit(features, outcomes)
	if verbose:
		print("Training accuracy:", model.score(features, outcomes))
	return model

def predict_treatment_probability(labels, features):
	model = LogisticRegression(solver='liblinear')
	model.fit(features, labels)
	if verbose:
		print("Training accuracy:", model.score(features, labels))
	treatment_probability = model.predict_proba(features)[:,1]
	return treatment_probability

def load_simulated_data():
	sim_df = pd.read_csv(simulation_file, delimiter='\t')
	sim_df = sim_df.rename(columns={'index':'post_index'})
	return sim_df

def load_term_counts(path='../dat/reddit/'):
	return load_npz(path + 'term_counts.npz').toarray()

def main():
	if not dat_dir:
		term_counts = load_term_counts()
	else:
		term_counts = load_term_counts(path=dat_dir)

	sim_df = load_simulated_data()
	treatment_labels = sim_df.treatment.values
	indices = sim_df.post_index.values
	all_words = term_counts[indices, :]

	treated_sim = sim_df[sim_df.treatment==1]
	untreated_sim = sim_df[sim_df.treatment==0]
	treated_indices = treated_sim.post_index.values
	untreated_indices = untreated_sim.post_index.values
	
	all_outcomes = sim_df.outcome.values
	outcomes_st_treated = treated_sim.outcome.values
	outcomes_st_not_treated = untreated_sim.outcome.values
	
	words_st_treated = term_counts[treated_indices,:]
	words_st_not_treated = term_counts[untreated_indices,:]

	treatment_probability = predict_treatment_probability(treatment_labels, all_words)
	model_outcome_st_treated = fit_conditional_expected_outcomes(outcomes_st_treated, words_st_treated)
	model_outcome_st_not_treated = fit_conditional_expected_outcomes(outcomes_st_not_treated, words_st_not_treated)

	expected_outcome_st_treated = predict_expected_outcomes(model_outcome_st_treated, all_words)
	expected_outcome_st_not_treated = predict_expected_outcomes(model_outcome_st_not_treated, all_words)

	q_hat = psi_q_only(expected_outcome_st_not_treated, expected_outcome_st_treated, 
			treatment_probability, treatment_labels, all_outcomes, truncate_level=0.03)

	tmle = psi_tmle_cont_outcome(expected_outcome_st_not_treated, expected_outcome_st_treated, 
			treatment_probability, treatment_labels, all_outcomes, truncate_level=0.03)[0]
	
	print("Q hat:", q_hat)
	print("TMLE:", tmle)


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