import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

import bert.tokenization as tokenization
from reddit.data_cleaning.reddit_posts import subreddit_idx_to_subreddit
from reddit.dataset.dataset import make_input_fn_from_file
from reddit.dataset.dataset import make_subreddit_based_simulated_labeler, make_propensity_based_simulated_labeler
from result_processing.helpers import convert_str_columns_to_float
from semi_parametric_estimation.att import att_estimates
from semi_parametric_estimation.helpers import calibrate_g


def subreddit_sim_ground_truth_and_naive_from_dataset(
        beta0=1.0,
        beta1=1.0,
        gamma=1.0,
        setting='simple',
        tfrecord_file='../dat/reddit/proc.tf_record',
        vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
        seed=0
):
    """
    This is a helper function that computes the ground truth and naive estimates
    by creating a tf dataset and 'sampling' the required observed data.
    This is necessary because we simulate outcomes on the fly using tf's dataset abstraction.


    :return:
    """

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    labeler = make_subreddit_based_simulated_labeler(treat_strength=beta0, con_strength=beta1, noise_level=gamma,
                                                     setting=setting, seed=seed)

    input_fn = make_input_fn_from_file(
        input_files_or_glob=tfrecord_file,
        seq_length=128,
        num_splits=10,
        dev_splits=0,
        test_splits=0,
        tokenizer=tokenizer,
        is_training=True,
        subreddits=[13, 6, 8],
        shuffle_buffer_size=int(1e6),  # note: bert hardcoded this, and I'm following suit
        seed=seed,
        labeler=labeler)

    # input_fn = make_input_fn_from_tfrecord(tokenizer=tokenizer, tfrecord=tfrecord_file)

    params = {'batch_size': 2048}

    dataset = input_fn(params)
    sampler = dataset.make_one_shot_iterator()
    sample = sampler.get_next()

    with tf.Session() as sess:
        # confounding = sess.run(sample['confounding'])
        y, y0, y1, t = sess.run([sample['outcome'], sample['y0'], sample['y1'], sample['gender']])

    tf.reset_default_graph()

    ground_truth = y1[t == 1].mean() - y0[t == 1].mean()
    very_naive = y1[t == 1].mean() - y0[t == 0].mean()

    return {'ground_truth': ground_truth, 'noiseless_very_naive': very_naive}


def propensity_sim_ground_truth_and_naive_from_dataset(
        beta0=1.0,
        beta1=1.0,
        gamma=1.0,
        exogenous_confounding=0.0,
        setting='simple',
        tfrecord_file='../dat/reddit/proc.tf_record',
        vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
        base_propensities_path='~/reddit_logs/prop_sim/propensity_source/test_results_keto.tsv',
        seed=0
):
    """
    This is a helper function that computes the ground truth and naive estimates
    by creating a tf dataset and 'sampling' the required observed data.
    This is necessary because we simulate outcomes on the fly using tf's dataset abstraction.

    :return:
    """

    output = pd.read_csv(base_propensities_path, '\t')
    base_propensity_scores = output['treatment_probability'].values
    # print(base_propensity_scores)
    example_indices = output['index'].values

    labeler = make_propensity_based_simulated_labeler(treat_strength=beta0,
                                                      con_strength=beta1,
                                                      noise_level=gamma,
                                                      base_propensity_scores=base_propensity_scores,
                                                      example_indices=example_indices,
                                                      exogeneous_con=exogenous_confounding,
                                                      setting=setting,
                                                      seed=seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    input_fn = make_input_fn_from_file(
        input_files_or_glob=tfrecord_file,
        seq_length=128,
        num_splits=10,
        dev_splits=0,
        test_splits=0,
        tokenizer=tokenizer,
        is_training=True,
        subreddits=[13],
        shuffle_buffer_size=int(1e6),  # note: bert hardcoded this, and I'm following suit
        seed=seed,
        labeler=labeler)

    # input_fn = make_input_fn_from_tfrecord(tokenizer=tokenizer, tfrecord=tfrecord_file)

    params = {'batch_size': 2048}

    dataset = input_fn(params)
    sampler = dataset.make_one_shot_iterator()
    sample = sampler.get_next()

    with tf.Session() as sess:
        # confounding = sess.run(sample['confounding'])
        y, y0, y1, t = sess.run([sample['outcome'], sample['y0'], sample['y1'], sample['treatment']])

    tf.reset_default_graph()

    ground_truth = y1[t == 1].mean() - y0[t == 1].mean()
    very_naive = y1[t == 1].mean() - y0[t == 0].mean()

    return {'ground_truth': ground_truth, 'noiseless_very_naive': very_naive}


def att_from_bert_tsv(tsv_path):
    output = pd.read_csv(tsv_path, '\t')
    output = convert_str_columns_to_float(output)

    y = output['outcome'].values
    t = output['treatment'].values
    q_t0 = output['expected_outcome_st_no_treatment'].values
    q_t1 = output['expected_outcome_st_treatment'].values
    g = output['treatment_probability'].values
    in_test = output['in_test'].values == 1
    in_train = np.logical_not(in_test)

    # g = calibrate_g(g, t)

    q_t0_test = q_t0[in_test]
    q_t1_test = q_t1[in_test]
    g_test = g[in_test]
    t_test = t[in_test]
    y_test = y[in_test]

    # all_estimates = att_estimates(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.03, prob_t=t.mean())
    # all_estimates = att_estimates(q_t0, q_t1, g, t, y, truncate_level=0.03, prob_t=t.mean(), deps=0.00001)
    all_estimates = att_estimates(q_t0[in_train], q_t1[in_train], g[in_train], t[in_train], y[in_train],
                                  truncate_level=0.03, prob_t=t.mean(), deps=0.00001)

    return all_estimates


def bert_psi(output_dir):
    """
    Expects that the data was split into k folds, and the predictions from each fold
    was saved in [output_dir]/[fold_identifier]/[output_name].
    (matches {}/*/*.tsv'.format(output_dir))

    :param output_dir:
    :return:
    """

    data_files = sorted(glob.glob('{}/*/*.tsv'.format(output_dir)))
    data_files += sorted(glob.glob('{}/*/predict/*.tsv'.format(output_dir), recursive=True))
    estimates = []
    for data_file in data_files:
        try:
            all_estimates = att_from_bert_tsv(data_file)
            # print(psi_estimates)
            estimates += [all_estimates]
        except:
            print('wtf')
            print(data_file)

    avg_estimates = {}
    for k in all_estimates.keys():
        k_estimates = []
        for estimate in estimates:
            k_estimates += [estimate[k]]

        avg_estimates[k] = np.mean(k_estimates)
        avg_estimates[(k, 'std')] = np.std(k_estimates)

    return avg_estimates


def process_subreddit_experiment(
        base_dir='logs/reddit_logs/simulated_training_linear_treatment/'):
    # for setting in ['simple', 'multiplicative', 'interaction']:
    outputs = []
    for setting in ['simple']:
        setting_dir = os.path.join(base_dir, "mode" + setting)
        print('---------------------------------')
        print(setting)
        print('---------------------------------')
        for gamma in [1.0, 4.0]:
            print('gamma: {}'.format(gamma))
            for beta1 in [1.0, 10.0, 100.0]:
                print("beta1: {}".format(beta1))
                # raw_and_gt = subreddit_sim_ground_truth_and_naive_from_dataset(
                #     beta0=1.0,
                #     beta1=beta1,
                #     gamma=gamma,
                #     setting=setting,
                #     seed=0)
                # print(raw_and_gt)
                output_dir = os.path.join(setting_dir, 'beta01.0.beta1{}.gamma{}'.format(beta1, gamma))
                # print(output_dir)
                estimates = bert_psi(output_dir)
                setting_info = {'setting': setting, 'gamma': gamma, 'beta1': beta1, 'ground_truth': 1.00}
                outputs += [{**setting_info, **estimates}]
        return outputs


def process_propensity_experiment(base_dir='logs/reddit_logs/prop_sim/'):
    # for setting in ['simple', 'multiplicative', 'interaction']:
    for setting in ['simple']:
        setting_dir = os.path.join(base_dir, "mode" + setting)
        print('---------------------------------')
        print(setting)
        print('---------------------------------')
        outputs = []
        for exog in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            print("exog: {}".format(exog))
            # raw_and_gt = propensity_sim_ground_truth_and_naive_from_dataset(
            #     beta0=1.0,
            #     beta1=10.0,
            #     gamma=0.0,
            #     exogenous_confounding=0.0,
            #     setting=setting,
            #     tfrecord_file='../dat/reddit/proc.tf_record',
            #     vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
            #     base_propensities_path='~/reddit_logs/prop_sim/propensity_source/test_results_keto.tsv',
            #     seed=0
            # )
            # print(raw_and_gt)
            output_dir = os.path.join(setting_dir, 'beta01.0.beta110.0.exog{}'.format(exog))
            # print(output_dir)
            estimates = bert_psi(output_dir)
            params = {'exog': exog, 'ground_truth': 1.00}
            outputs += [{**params, **estimates}]
        return outputs


def process_reddit_application(base_dir='logs/reddit_logs/real/seed0'):
    # for setting in ['simple', 'multiplicative', 'interaction']:
    outputs = []

    for subreddit in [6, 8, 13]:
        output_dir = os.path.join(base_dir, "subreddit" + str(subreddit))
        print('---------------------------------')
        print(subreddit_idx_to_subreddit(subreddit))
        print('---------------------------------')
        estimates = bert_psi(output_dir)
        params = {'subreddit': subreddit_idx_to_subreddit(subreddit)}
        outputs += [{**params, **estimates}]

    return outputs


def main():
    # tf.enable_eager_execution()

    # gt_and_naive = subreddit_sim_ground_truth_and_naive_from_dataset(
    #         beta0=1.0,
    #         beta1=1.0,
    #         gamma=1.0,
    #         setting='interaction',
    #         tfrecord_file='../dat/reddit/proc.tf_record',
    #         vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
    #         seed=0
    # )
    # print(gt_and_naive)

    # gt_and_naive = propensity_sim_ground_truth_and_naive_from_dataset(
    #     beta0=1.0,
    #     beta1=10.0,
    #     gamma=0.0,
    #     exogenous_confounding=1.0,
    #     setting='simple',
    #     tfrecord_file='../dat/reddit/proc.tf_record',
    #     vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
    #     base_propensities_path='~/reddit_logs/prop_sim/propensity_source/test_results_keto.tsv',
    #     seed=0
    # )
    # print(gt_and_naive)

    print("*****************************************")
    print("Subreddit as Confounding experiment")
    print("*****************************************")
    outputs = pd.DataFrame(process_subreddit_experiment())
    print(outputs[['gamma', 'beta1', 'very_naive', 'plugin', 'one_step_tmle']])

    # outputs = pd.DataFrame(process_propensity_experiment())
    # print(outputs[['exog', 'very_naive', 'plugin', 'one_step_tmle']])

    print("*****************************************")
    print("Reddit Application")
    print("*****************************************")
    outputs = pd.DataFrame(process_reddit_application())
    print(outputs[['subreddit', 'very_naive', 'plugin', 'one_step_tmle']])
    print(outputs.loc[:, [('very_naive', 'std'), ('plugin', 'std'), ('one_step_tmle', 'std')]])


if __name__ == '__main__':
    main()
    # pass
