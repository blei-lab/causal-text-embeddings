import os
import argparse
from reddit.data_cleaning.reddit_posts import load_term_counts
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from scipy.stats import chi2, ttest_1samp
from .helpers import convert_str_columns_to_float, assign_split


#t ~ Bernoulli(sigmoid(bW + aZ + b0))
def fit_treatment_model(df, term_counts):
	indices = df.post_index.values
	tc = term_counts[indices,:]
	tc = tc.toarray()
	f_z = logit(df.treatment_probability.values)
	print(f_z.shape, tc.shape)
	features = np.column_stack((f_z, tc))
	labels = df.treatment.values

	true_model = LogisticRegression(solver='liblinear')
	true_model.fit(features, labels)
	coeffs = np.array(true_model.coef_).flatten()[1:]
	print(coeffs.mean(), coeffs.std())

	np.random.shuffle(tc)
	features = np.column_stack((f_z, tc))
	permuted = LogisticRegression(solver='liblinear')
	permuted.fit(features, labels)
	permuted_coeffs = np.array(permuted.coef_).flatten()[1:]
	print(permuted_coeffs.mean(), permuted_coeffs.std())


#$E_{Z|W=1}[log P(T=1 | W=1, Z)/ P(T=1| Z)]$
def compute_expected_treatment_likelihood_ratio(df, term_counts, word_index, do_shuffle=False):
	word_occurence = term_counts[:,word_index].toarray()
	mask_word = (word_occurence == 1).flatten()
	p_ind = np.arange(term_counts.shape[0])
	n = p_ind[mask_word].shape[0]
	if do_shuffle:
		np.random.shuffle(p_ind)
		rand_indices = p_ind[:mask_word.shape[0]]
		df_word_occurs = df[df.post_index.isin(rand_indices)]
	else:
		df_word_occurs = df[df.post_index.isin(p_ind[mask_word])]
	
	model_p_z = train_treatment_classifier(df, term_counts, word_index, occurence=None)
	prob_t_given_z = predict_treatment_prob_from_model(model_p_z, df_word_occurs)
	model_st_word = train_treatment_classifier(df_word_occurs, term_counts, word_index,occurence=None)
	prob_st_word = predict_treatment_prob_from_model(model_st_word, df_word_occurs)
	log_ratio = np.log(prob_st_word/prob_t_given_z)
	print("Mean (and std err.) of ratio:", log_ratio.mean(), log_ratio.std()/np.sqrt(n))
	print("N = ", n)
	return ttest_1samp(log_ratio, 0.0)


def predict_treatment_prob_from_model(model, df):
	features = logit(df.treatment_probability.values)
	features = features[:,np.newaxis]
	return model.predict_proba(features)[:,1]

def compute_treatment_likelihood_ratio(df, term_counts, word_index):
	null_model = train_treatment_classifier(df, term_counts, word_index, occurence=None)
	model_st_word = train_treatment_classifier(df, term_counts, word_index)
	model_st_no_word = train_treatment_classifier(df, term_counts, word_index, occurence=0)

	word_occurence = term_counts[:,word_index].toarray()
	mask_word = (word_occurence == 1).flatten()
	mask_no_word = (word_occurence == 0).flatten()
	p_ind = np.arange(term_counts.shape[0])
	word_occurs = df[df.post_index.isin(p_ind[mask_word])]
	word_no_occurs = df[df.post_index.isin(p_ind[mask_no_word])]

	null_given_word = predict_treatment_prob_from_model(null_model, word_occurs)
	null_given_no_word = predict_treatment_prob_from_model(null_model, word_no_occurs)
	p_t_given_word = predict_treatment_prob_from_model(model_st_word, word_occurs)
	p_t_given_no_word = predict_treatment_prob_from_model(model_st_no_word, word_no_occurs)

	ratio_word = np.log(null_given_word / p_t_given_word)
	ratio_no_word = np.log(null_given_no_word / p_t_given_no_word)
	ratio = np.hstack([ratio_word, ratio_no_word])
	print("Mean and std of log likelihood ratios", ratio.mean(), ratio.std())
	# return -2*ratio.sum(), chi2.pdf(-2*ratio.sum(), df=2)
	return ttest_1samp(ratio, 0.0)

def train_treatment_classifier(df, term_counts, word_index, occurence=1):
	if occurence is not None:
		term_counts = term_counts[:,word_index].toarray()
		post_indices = np.arange(term_counts.shape[0])
		mask = (term_counts==occurence).flatten()
		post_indices = post_indices[mask]
		df = df[df.post_index.isin(post_indices)]
	
	labels = df.treatment.values
	features = logit(df.treatment_probability.values)
	features = features[:,np.newaxis]
	model = LogisticRegression(solver='liblinear')
	model.fit(features, labels)
	return model

def likelihood_ratio_hypothesis_test(df, term_counts, word_index):
	treated = df[df.treatment==1]
	control = df[df.treatment==0]
	null_model = train_classifier(df, term_counts, word_index, treat_index=None)
	model_st_treated = train_classifier(df, term_counts, word_index)
	model_st_not_treated = train_classifier(df, term_counts, word_index, treat_index=0)

	null_st_treated = compute_word_occur_prob(null_model, treated)
	null_st_not_treated = compute_word_occur_prob(null_model, control)
	prob_st_treated = compute_word_occur_prob(model_st_treated, treated)
	prob_st_not_treated = compute_word_occur_prob(model_st_not_treated)

	ratio_treated = np.log(null_st_treated / prob_st_treated)
	ratio_control = np.log(null_st_not_treated / prob_st_not_treated)
	ratios = np.hstack([ratio_treated,ratio_control])

	print("Mean and std. of log likelihood ratios:", ratios.mean(), ratios.std())
	return ttest_1samp(ratios, 0.0)
	
	# statistic = -2 * ratios.sum()
	# p_value = chi2.pdf(statistic, df=2)
	# return statistic, p_value

def compute_unadjusted_statistic(test_df, term_counts, word_index):
	term_counts = term_counts[:,word_index]
	term_counts[term_counts > 1] = 1
	treated_indices = test_df.loc[test_df.treatment == 1, 'post_index'].values
	control_indices = test_df.loc[test_df.treatment == 0, 'post_index'].values
	treated_term_frequency = term_counts[treated_indices].sum()/treated_indices.shape[0]
	control_term_frequency = term_counts[control_indices].sum()/control_indices.shape[0]
	stat = np.log(treated_term_frequency) - np.log(control_term_frequency)
	return abs(stat)

def compute_conditional_likelihood_ratios(df, term_counts, word_index):
	model_st_treated = train_classifier(df, term_counts, word_index)
	model_st_not_treated = train_classifier(df, term_counts, word_index, treat_index=0)
	probability_st_treated = compute_word_occur_prob(model_st_treated, df)
	probability_st_not_treated = compute_word_occur_prob(model_st_not_treated, df)
	log_likelihood_ratio = np.log(probability_st_treated / probability_st_not_treated)
	print("Mean and std. of log ll ratio", log_likelihood_ratio.mean(), ";", log_likelihood_ratio.std())
	print("sanity check:", (log_likelihood_ratio.mean()/(log_likelihood_ratio.std()/np.sqrt(df.shape[0]))))
	return ttest_1samp(log_likelihood_ratio, 0.0)

def compute_probabilities(model, test_df, term_counts, word_index):
	indices = test_df.post_index.values
	word_label = term_counts[indices, word_index].toarray().flatten()
	word_label[word_label>1]=1
	features = logit(test_df.treatment_probability.values)
	features = features[:,np.newaxis]
	probs = model.predict_proba(features)[:,word_label] 
	return probs.flatten()

def compute_word_occur_prob(model, test_df):
	features = logit(test_df.treatment_probability.values)
	features = features[:,np.newaxis]
	return model.predict_proba(features)[:,1] 
	
def get_author_frequency(author_term_counts):
	return author_term_counts.sum(axis=0)

def get_nonnumeric_term_indices(vocab):
	return set([idx for idx in range(vocab.shape[0]) if vocab[idx].isalpha()])

def get_top_words(df, term_counts, author_term_counts, vocab, n=10, use_author_freq_weight=False):
	indices = df.post_index.values
	term_counts = term_counts[indices, :]
	valid_word_indices = get_nonnumeric_term_indices(vocab)
	term_counts = np.array(term_counts.sum(axis=0))
	
	if use_author_freq_weight:
		author_freq = get_author_frequency(author_term_counts)
		term_counts = term_counts / np.array(author_freq)

	sorted_tc = np.argsort(term_counts)
	top_indices = []
	idx = 0
	while len(top_indices) < n:
		cand_term = sorted_tc[0,-idx]
		if cand_term in valid_word_indices:
			top_indices.append(cand_term)
		idx += 1
	return top_indices

def train_classifier(train_df, term_counts, word_index, treat_index=1):
	if treat_index is not None: 
		train_df = train_df[train_df.treatment==treat_index]

	indices = train_df.post_index.values
	term_counts = term_counts[:,word_index]
	labels = term_counts[indices,:]
	labels = labels.toarray().flatten()
	labels[labels>1] = 1
	features = logit(train_df.treatment_probability.values)
	features = features[:,np.newaxis]
	model = LogisticRegression(solver='liblinear')
	model.fit(features, labels)
	return model

def truncate(df, truncate_level=0.1):
	df = df[(df.treatment_probability >= truncate_level) & (df.treatment_probability <= 1.0-truncate_level)]
	return df


def do_fit():
	predictions_file = os.path.join(log_dir, 'predict', 'test_results_all.tsv')
	predict_df = pd.read_csv(predictions_file, delimiter='\t')
	predict_df = convert_str_columns_to_float(predict_df)
	predict_df = predict_df.rename(columns={'index':'post_index'})
	# predict_df = truncate(predict_df)
	term_counts, author_term_counts, vocab = load_term_counts(force_redo=redo_proc)
	fit_treatment_model(predict_df, term_counts)


def main():
	predictions_file = os.path.join(log_dir, 'predict', 'test_results_all.tsv')
	predict_df = pd.read_csv(predictions_file, delimiter='\t')
	predict_df = convert_str_columns_to_float(predict_df)
	predict_df = predict_df.rename(columns={'index':'post_index'})
	predict_df = truncate(predict_df)
	predict_df = assign_split(predict_df, num_splits=2)

	n_words = 10
	term_counts, author_term_counts, vocab = load_term_counts(force_redo=redo_proc)
	top_word_indices = get_top_words(predict_df, term_counts, author_term_counts, vocab, n=n_words, use_author_freq_weight=False)
	
	for idx in range(n_words):
		word_idx = top_word_indices[idx]
		print("Working on word:", vocab[word_idx], "(index", idx, ")")

		(statistic, p_value) = compute_expected_treatment_likelihood_ratio(predict_df, term_counts, word_idx)
		print("t-statistic and p-value:", statistic, ";", p_value)
		print("--"*40)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--log-dir", action="store", default="../logdir/beta04.0.beta110.0.gamma1.0")
	parser.add_argument("--redo-proc", action="store_true")
	args = parser.parse_args()
	log_dir = args.log_dir
	redo_proc = args.redo_proc

	# do_fit()
	main()