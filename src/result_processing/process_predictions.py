import os
import argparse
from reddit.data_cleaning import load_reddit, process_text_length
import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from .helpers import convert_str_columns_to_float


def plot_covariate_proportions_per_stratum(treated, control, num_bins, covariate='subreddit'):
	cov_vals = treated[covariate].values
	n_groups = num_bins
	
	for val in cov_vals:
		# data to plot
		treat_props = treated.loc[treated[covariate] == val, 'count'].values
		control_props = control.loc[control[covariate] == val, 'count'].values

		# create plot
		fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 0.3
		opacity = 0.8

		rects1 = plt.bar(index, treat_props, bar_width,
		alpha=opacity,
		color='b',
		label='Treated Units')

		rects2 = plt.bar(index + bar_width, control_props, bar_width,
		alpha=opacity,
		color='g',
		label='Control Units')

		plt.ylim((0.0,1.0))
		plt.xlabel('Stratas')
		plt.ylabel('Proportions of posts in ' + covariate + ':' + val)
		plt.xticks(index + bar_width, tuple(range(1,num_bins+1)))
		plt.legend()

		plt.tight_layout()
		plt.savefig(os.path.join(log_dir, 'proportions_for_' + covariate + '_' + val + '.png'))

def normalize(df, col):
	vals = df[col].values
	min_col = vals.min()
	max_col = vals.max()
	df[col] = (df[col] - min_col)/(max_col-min_col)
	return df


def get_covariate_proportions(stratified_df, covariate='subreddit'):
	counts_df = stratified_df.groupby(['strata', covariate]).size().reset_index(name="count")
	total_by_strata = stratified_df.groupby("strata").size().reset_index(name="total")
	counts_df = counts_df.merge(total_by_strata, how='inner', on='strata')
	counts_df['count'] /= counts_df['total']
	return counts_df


def get_text_results(reddit_df, result_df, sub=None):
	indices = result_df['index'].values
	result_df = reddit_df.loc[indices, ['subreddit', 'post_text', 'author']]

	if sub:
		result_df = result_df[result_df.subreddit.isin([sub])]

	return result_df


def print_example_posts(sub_text_df, n=10):
	post_list = [tuple(val) for val in sub_text_df.values]
	random_posts = sample(post_list, n)
	print("*"*10 + "Examples" + "*"*10)
	for post in random_posts:
		print("Subreddit:", post[0])
		print("-"*40)
		print("Text:", post[1])
		print("-"*40)
		print("Author:", post[2])
		print("*"*40)


def stratify_by_value(df, num_bins=10, sort_by='treatment_probability', col_to_add='strata'):
	values = df[sort_by].values
	min_val = values.min()
	max_val = values.max()
	interval = (max_val-min_val)/num_bins
	bins = np.arange(min_val, max_val, step=interval)
	bin_indices = np.digitize(values, bins)
	df[col_to_add] = bin_indices
	return df


def main():
	num_examples_to_print=5
	num_bins = 5

	predictions_file = os.path.join(log_dir, 'predict', 'test_results_all.tsv')
	predict_df = pd.read_csv(predictions_file, delimiter='\t')
	predict_df = convert_str_columns_to_float(predict_df)
	predict_df = predict_df.rename(columns={'index':'post_index'})
	print(predict_df)

	treated = predict_df[predict_df.treatment == 1]
	control = predict_df[predict_df.treatment == 0]

	treated_stratified = stratify_by_value(treated, num_bins=num_bins)
	control_stratified = stratify_by_value(control, num_bins=num_bins)

	if res_type == 'subreddit':
		treated_cov_prop = get_covariate_proportions(treated_stratified)
		control_cov_prop = get_covariate_proportions(control_stratified)

		plot_covariate_proportions_per_stratum(treated_cov_prop, control_cov_prop, num_bins)

		for i in range(1,num_bins+1):
			print("*"*20, "Proportions for stratum:", i, "*"*20)
			print("-"*10, "Treated:", "-"*10)
			print(treated_cov_prop[treated_cov_prop.strata == i])

			print("-"*10, "Control:", "-"*10)
			print(control_cov_prop[control_cov_prop.strata == i])

	elif res_type == 'length':
		text = load_reddit()
		text = process_text_length(text)
		text = normalize(text, 'post_length')

		treated = treated.merge(text, left_on='post_index', right_index=True, how='inner')
		control = control.merge(text, left_on='post_index', right_index=True, how='inner')

		treated_corr = pearsonr(treated.post_length.values, treated.treatment_probability.values)
		control_corr = pearsonr(control.post_length.values, control.treatment_probability.values)
		print("Corr. between treated and post length", treated_corr)
		print("Corr. between control and post length", control_corr)


		# binned_post_length = stratify_by_value(text, num_bins=20, sort_by='post_length', col_to_add='length_bin')

		# columns_to_keep = treated_stratified.columns.tolist().extend('length_bin')
		# treated_text = treated_stratified.merge(binned_post_length, left_on='post_index', right_index=True, how='inner')# [columns_to_keep]
		# control_text = control_stratified.merge(binned_post_length, left_on='post_index', right_index=True, how='inner')#[columns_to_keep]

		# treated_cov_prop = get_covariate_proportions(treated_text, covariate='length_bin')
		# control_cov_prop = get_covariate_proportions(control_text, covariate='length_bin')

		# for i in range(1,num_bins+1):
		# 	print("*"*20, "Proportions for stratum:", i, "*"*20)
		# 	print("-"*10, "Treated:", "-"*10)
		# 	print(treated_cov_prop[treated_cov_prop.strata == i])

		# 	print("-"*10, "Control:", "-"*10)
		# 	print(control_cov_prop[control_cov_prop.strata == i])



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--log-dir", action="store", default="../logdir/simulated_training_1.0_1.0_1.0")
	parser.add_argument("--result-type", action="store", default="subreddit")
	args = parser.parse_args()
	log_dir = args.log_dir
	res_type = args.result_type

	main()