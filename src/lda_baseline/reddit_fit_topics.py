from reddit.data_cleaning.reddit_posts import load_reddit
from .helpers import tokenize_documents, assign_dev_split, learn_topics, show_topics, filter_by_subreddit
import numpy as np
import pandas as pd
import os
from scipy import sparse
import argparse
import sys

def load_term_counts(reddit, path='../dat/reddit/', force_redo=False):
	count_filename = path  + 'term_counts'
	vocab_filename = path + 'vocab'

	if os.path.exists(count_filename + '.npz') and not force_redo:
		return sparse.load_npz(count_filename + '.npz'), np.load(vocab_filename + '.npy')

	post_docs = reddit['post_text'].values
	counts, vocab, _ = tokenize_documents(post_docs)    
	sparse.save_npz(count_filename, counts)
	np.save(vocab_filename, vocab)
	return counts, np.array(vocab)

def main():
	if not os.path.exists(os.path.join(out_dir, 'topics.npy')) or redo_lda:

		subreddits = {'keto', 'OkCupid', 'childfree'}
		reddit = load_reddit()
		filtered_indices = filter_by_subreddit(reddit, subs=subreddits)

		if dat_dir:
			terms, vocab = load_term_counts(reddit, path=dat_dir, force_redo=redo_proc)
		else:
			terms, vocab = load_term_counts(reddit, force_redo=redo_proc)

		terms = terms[filtered_indices, :]
		N = terms.shape[0]
		indices = np.arange(N)
		dev_idx = assign_dev_split(N)
		train_idx = np.setdiff1d(indices, dev_idx)
		X_tr = terms[train_idx, :]
		X_dev = terms[dev_idx, :]
		print(dev_idx.shape)

		K_vals = [100]
		validation_scores = np.zeros(len(K_vals))
		all_topics = []
		models = []
		for i,k in enumerate(K_vals):
			score, lda_obj, topics = learn_topics(X_tr, X_dev, K=k)
			validation_scores[i] = score
			all_topics.append(topics)
			models.append(lda_obj)
		k_idx = np.argsort(validation_scores)[0]#[-1]
		best_k = K_vals[k_idx]
		best_topics = all_topics[k_idx]
		best_model = models[k_idx] 
		best_doc_prop = best_model.transform(terms)
		np.save(os.path.join(out_dir, 'topics'), best_topics)
		np.save(os.path.join(out_dir, 'document_proportions'), best_doc_prop)
	else:
		best_topics = np.load(os.path.join(out_dir, 'topics.npy'))
		vocab = np.load(os.path.join(out_dir, 'vocab.npy'))

	# print("Best topic")
	# topics = show_topics(vocab, best_topics, n_words=10)
	# print(topics)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dat-dir", action="store", default=None)
	parser.add_argument("--out-dir", action="store", default="../dat/reddit/")
	parser.add_argument("--redo-lda", action="store_true")
	parser.add_argument("--redo-proc", action="store_true")
	parser.add_argument("--test", action="store_true")
	args = parser.parse_args()
	out_dir = args.out_dir
	redo_lda = args.redo_lda
	redo_proc = args.redo_proc
	dat_dir = args.dat_dir
	test = args.test

	main()