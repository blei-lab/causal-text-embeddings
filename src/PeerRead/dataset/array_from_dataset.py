"""
helpers to take samples from the dataset and turn them into numpy arrays
(for ease of inspection and use with baselines)
"""
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import os
try:
    import mkl_random as random
except ImportError:
    import numpy.random as random

import bert.tokenization as tokenization
from PeerRead.dataset.dataset import make_input_fn_from_file, make_buzzy_based_simulated_labeler


def dataset_fn_to_df(dataset_fn):

    params = {'batch_size': 1}
    dataset = dataset_fn(params)

    itr = dataset.make_one_shot_iterator()

    samples = []

    for i in range(25000):
        try:
            sample = itr.get_next()
            for k in sample:
                sample[k] = sample[k].numpy()[0]
            samples += [sample]
            # print("year: {}".format(sample['year']))
        except:
            print(i)
            break

    df = pd.DataFrame(samples)

    return df

def buzzy_title_based_sim_dfs(treat_strength, con_strength, noise_level, setting="simple", seed=0,
                            base_output_dir='../dat/sim/peerread_buzzytitle_based/'):

    labeler = make_buzzy_based_simulated_labeler(treat_strength, con_strength, noise_level, setting=setting, seed=seed)

    num_splits = 10
    dev_splits = [0]
    test_splits = [0]

    # data_file = '../dat/reddit/proc.tf_record'
    # vocab_file = "../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    input_dataset_from_filenames = make_input_fn_from_file(data_file,
                                                           250,
                                                           num_splits,
                                                           dev_splits,
                                                           test_splits,
                                                           tokenizer,
                                                           is_training=False,
                                                           filter_test=False,
                                                           shuffle_buffer_size=25000,
                                                           seed=seed,
                                                           labeler=labeler)

    output_df = dataset_fn_to_df(input_dataset_from_filenames)
    output_df = output_df.rename(index=str, columns={'theorem_referenced': 'treatment'})

    output_dir = os.path.join(base_output_dir, "mode{}".format(setting))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "beta0{}.beta1{}.gamma{}.tsv".format(treat_strength, con_strength, noise_level))

    output_df.to_csv(output_path, '\t')


def main():
    tf.enable_eager_execution()

    buzzy_title_based_sim_dfs(treat_strength=beta0, con_strength=beta1, noise_level=gamma, setting=mode, seed=0,
                            base_output_dir=base_output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", action="store", default='../dat/PeerRead/proc/arxiv-all.tf_record')
    parser.add_argument("--vocab-file", action="store", default='../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt')
    parser.add_argument("--base-output-dir", action="store", default='../dat/sim/peerread_buzzytitle_based/')
    parser.add_argument("--mode", action="store", default="simple")
    parser.add_argument("--beta0", action="store", default='1.0')
    parser.add_argument("--beta1", action="store", default='1.0')
    parser.add_argument("--gamma", action="store", default='1.0')
    args = parser.parse_args()

    data_file = args.data_file
    vocab_file = args.vocab_file
    base_output_dir = args.base_output_dir
    mode = args.mode
    beta0 = float(args.beta0)
    beta1 = float(args.beta1)
    gamma = float(args.gamma)

    main()