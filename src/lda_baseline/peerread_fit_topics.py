from .helpers import tokenize_documents, assign_dev_split, learn_topics, show_topics, filter_by_subreddit
import numpy as np
import pandas as pd
import os
from scipy import sparse
import argparse
import sys

def load_peerread(path='../dat/PeerRead/'):
	return pd.read_csv(path + 'proc_abstracts.csv')


def load_term_counts(df, path='../dat/PeerRead/', force_redo=False, text_col='abstract_text'):
	count_filename = path  + 'term_counts'
	vocab_filename = path + 'vocab'

	if os.path.exists(count_filename + '.npz') and not force_redo:
		return sparse.load_npz(count_filename + '.npz'), np.load(vocab_filename + '.npy')

	post_docs = df[text_col].values
	counts, vocab, _ = tokenize_documents(post_docs)    
	sparse.save_npz(count_filename, counts)
	np.save(vocab_filename, vocab)
	return counts, np.array(vocab)

def main():
	if not os.path.exists(os.path.join(out_dir, 'topics.npy')) or redo_lda:
		if dat_dir:
			peerread = load_peerread(path=dat_dir)
			terms, vocab = load_term_counts(peerread, path=dat_dir, force_redo=redo_proc)
		else:
			peerread = load_peerread()
			terms, vocab = load_term_counts(peerread, force_redo=redo_proc)

		N = terms.shape[0]
		indices = np.arange(N)
		dev_idx = assign_dev_split(N)
		train_idx = np.setdiff1d(indices, dev_idx)
		X_tr = terms[train_idx, :]
		X_dev = terms[dev_idx, :]
		K_vals = [50]
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

	print("Best topic")
	topics = show_topics(vocab, best_topics, n_words=10)
	print(topics)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dat-dir", action="store", default=None)
	parser.add_argument("--out-dir", action="store", default="../dat/PeerRead/")
	parser.add_argument("--redo-lda", action="store_true")
	parser.add_argument("--redo-proc", action="store_true")
	parser.add_argument("--test", action="store_true")
	args = parser.parse_args()
	out_dir = args.out_dir
	redo_lda = args.redo_lda
	redo_proc = args.redo_proc
	dat_dir = args.dat_dir

	main()