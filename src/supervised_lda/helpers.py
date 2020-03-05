from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, articles):
		stop = stopwords.words('english')
		return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t.isalpha() and t not in stop]

def filter_by_subreddit(reddit, subs=None):
	if not subs:
		return reddit.index.values
	else:
		return reddit[reddit.subreddit.isin(subs)].index.values

def tokenize_documents(documents,max_df0=0.9, min_df0=0.0005):
	from nltk.corpus import stopwords
	'''
	From a list of documents raw text build a matrix DxV
	D: number of docs
	V: size of the vocabulary, i.e. number of unique terms found in the whole set of docs
	'''
	count_vect = CountVectorizer(tokenizer=LemmaTokenizer(), max_df=max_df0, min_df=min_df0)
	corpus = count_vect.fit_transform(documents)
	vocabulary = count_vect.get_feature_names()
	
	return corpus,vocabulary,count_vect

def assign_dev_split(num_docs, percentage=0.05):
	indices = np.arange(num_docs)
	np.random.shuffle(indices)
	size = int(indices.shape[0]*percentage)
	dev = indices[:size]
	return dev

def learn_topics(X, X_dev, K=50):
	lda = LatentDirichletAllocation(n_components=K, learning_method='online', verbose=1)
	print("Fitting", K, "topics...")
	lda.fit(X)
	score = lda.perplexity(X_dev)
	print("Log likelihood:", score)
	topics = lda.components_
	return score, lda, topics

def show_topics(vocab, topics, n_words=20):
	topic_keywords = []
	for topic_weights in topics:
		top_keyword_locs = (-topic_weights).argsort()[:n_words]
		topic_keywords.append(vocab.take(top_keyword_locs))

	df_topic_keywords = pd.DataFrame(topic_keywords)
	df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
	df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
	return df_topic_keywords

def filter_document_embeddings(filtered_df, doc_embeddings, index_mapping, on='post_index'):
	filtered_indices = filtered_df[on].values
	doc_idx = [index_mapping[idx] for idx in filtered_indices]
	embeddings = doc_embeddings[doc_idx, :]
	return embeddings

def filter_document_terms(filtered_df, counts, index_mapping, on='post_index'):
	filtered_indices = filtered_df[on].values
	doc_idx = [index_mapping[idx] for idx in filtered_indices]
	filtered_counts = counts[doc_idx, :]
	return filtered_counts

def make_index_mapping(df, on='post_index', convert_to_int=True):
	if on=='index':
		indices = df.index.values
	else:
		indices = df[on].values

	if convert_to_int:
		return {int(ind):i for (i,ind) in enumerate(indices)}

	return {ind:i for (i,ind) in enumerate(indices)}

def assign_split(df, num_splits=10, col_to_add='split'):
	df[col_to_add] = np.random.randint(0, num_splits, size=df.shape[0])
	return df
