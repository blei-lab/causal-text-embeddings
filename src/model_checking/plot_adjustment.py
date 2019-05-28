import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.special import logit
from result_processing.helpers import convert_str_columns_to_float, assign_split, filter_imbalanced_terms
from sklearn.metrics import mean_squared_error as mse
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def get_prediction_file():
	predict_df = pd.read_csv(log_file, delimiter='\t')
	predict_df = predict_df.rename(columns={'index':'post_index'})
	return predict_df

def fit_treatment(features, labels, verbose=False, coeff_offset=1):
	model = LogisticRegression(solver='liblinear')
	model.fit(features, labels)
	coeffs = np.array(model.coef_).flatten()[coeff_offset:]
	if verbose:
		print("Model accuracy:", model.score(features, labels))
		print("Mean and std. of the word coeffs:", coeffs.mean(), coeffs.std())     
	return coeffs

def truncate(df, truncate_level=0.1):
	df = df[(df.treatment_probability >= truncate_level) & (df.treatment_probability <= 1.0-truncate_level)]
	return df

def plot_density(unadjusted, adjusted, permuted):
	density = gaussian_kde(adjusted.mean(axis=0))
	permutation_density = gaussian_kde(permuted.mean(axis=0))
	missing_z_density = gaussian_kde(unadjusted.mean(axis=0))
	xs = np.linspace(-0.5,0.5,1000)
	plt.plot(xs,density(xs), label='Adjusted model (not permuted)')
	plt.plot(xs, permutation_density(xs), label='Permuted model')
	plt.plot(xs, missing_z_density(xs), label='Unadjusted model')
	plt.xlabel('Coefficient values for words')
	plt.legend()

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	# plt.tight_layout()
	plt.savefig(out_dir + out_file, dpi=300)

def load_terms(data):
	termfile = '../dat/' + data + '/term_counts.npz'
	if data == 'reddit':
		termfile = '../dat/' + data + '_term_counts.npz'
	term_counts = load_npz(termfile).toarray()
	if drop_terms:
		term_indices = np.arange(term_counts.shape[1])
		random_indices = np.random.choice(term_indices, 1000)
		term_counts = term_counts[:,random_indices]
	return term_counts

def main():
	predict_df = get_prediction_file()
	term_counts = load_terms(dataset)
	print(predict_df.shape, term_counts.shape)
	if dataset == 'reddit':
		imbalanced_terms = filter_imbalanced_terms(predict_df, term_counts)
		term_counts = term_counts[:,imbalanced_terms]
		print(term_counts.shape)

	n_bootstraps = 10
	n_w = term_counts.shape[1]
	
	adjusted = np.zeros((n_bootstraps, n_w))
	permuted = np.zeros((n_bootstraps, n_w))
	unadjusted = np.zeros((n_bootstraps, n_w))

	for i in range(n_bootstraps):
		sample = assign_split(predict_df,num_splits=2)
		sample = sample[sample.split==0]
		indices = sample.post_index.values
		labels = sample.treatment.values
		words = term_counts[indices, :]
		propensity_score = logit(sample.treatment_probability.values)
		all_features = np.column_stack((propensity_score, words))
		unadjusted[i,:] = fit_treatment(words, labels, coeff_offset=0)
		adjusted[i,:] = fit_treatment(all_features, labels)
		np.random.shuffle(words)
		permuted_features = np.column_stack((propensity_score, words))
		permuted[i,:] = fit_treatment(permuted_features, labels)

	plot_density(unadjusted, adjusted, permuted)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--out-dir", action="store", default='../figures/')
	parser.add_argument("--out-file", action="store", default='reddit.pdf')
	parser.add_argument("--log-file", action="store", default='../logdir/reddit/modesimple/beta01.0.beta110.0.gamma1.0/predict/test_results_all.tsv')
	parser.add_argument("--drop-terms", action="store_true")
	parser.add_argument("--dataset", action="store", default='reddit')
	args = parser.parse_args()
	log_file = args.log_file
	drop_terms = args.drop_terms
	dataset = args.dataset
	out_dir = args.out_dir
	out_file = args.out_file
	main()