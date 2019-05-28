import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
np.random.seed(0)

def convert_str_columns_to_float(df):
	df['expected_outcome_st_treatment'] = df['expected_outcome_st_treatment'].str[1:-1]
	df['expected_outcome_st_treatment'] = df['expected_outcome_st_treatment'].astype(np.float64)

	df['expected_outcome_st_no_treatment'] = df['expected_outcome_st_no_treatment'].str[1:-1]
	df['expected_outcome_st_no_treatment'] = df['expected_outcome_st_no_treatment'].astype(np.float64)
	return df


def tokenize_documents(documents,max_df0=0.8, min_df0=0.01,print_vocabulary=False,outfolder=None,output_vocabulary_fname='vocabulary.dat'):
	from nltk.corpus import stopwords
	'''
	From a list of documents raw text build a matrix DxV
	D: number of docs
	V: size of the vocabulary, i.e. number of unique terms found in the whole set of docs
	'''
	stop = stopwords.words('english')
	count_vect = CountVectorizer(stop_words=stop,max_df=max_df0, min_df=min_df0)
	corpus = count_vect.fit_transform(documents)
	vocabulary = count_vect.get_feature_names()
	
	return corpus,vocabulary,count_vect


def assign_split(df, num_splits=10, col_to_add='split'):
	df[col_to_add] = np.random.randint(0, num_splits, size=df.shape[0])
	return df


def filter_imbalanced_terms(df, term_counts, imbalance=0.1, key='post_index'):
	t_indices = []
	n_terms = term_counts.shape[1]
	for t in range(n_terms):
		ind_occur = np.nonzero(term_counts[:,t])[0]
		subset = df[df[key].isin(ind_occur)]
		if subset.shape[0] != 0:
			prop_men = subset[subset.treatment==1].shape[0]/subset.shape[0]
			prop_women = subset[subset.treatment==0].shape[0]/subset.shape[0]
			if abs(prop_women-prop_men)>=imbalance:
				t_indices.append(t)
	return t_indices





