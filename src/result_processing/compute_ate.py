import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

import bert.tokenization as tokenization
from PeerRead.dataset.dataset import make_input_fn_from_file
from PeerRead.dataset.dataset import make_buzzy_based_simulated_labeler, make_propensity_based_simulated_labeler
from semi_parametric_estimation.ate import ate_estimates, ates_from_atts


def buzzy_sim_ground_truth_and_naive_from_dataset(
        beta0=0.25,
        beta1=0.0,
        gamma=0.0,
        setting='simple',
        tfrecord_file='../dat/PeerRead/proc/arxiv-all.tf_record',
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
    labeler = make_buzzy_based_simulated_labeler(treat_strength=beta0, con_strength=beta1, noise_level=gamma,
                                                 setting=setting, seed=seed)

    input_fn = make_input_fn_from_file(tfrecord_file,
                                     250,
                                     num_splits=10,
                                     dev_splits=[0],
                                     test_splits=[0],
                                     tokenizer=tokenizer,
                                     is_training=True,
                                     filter_test=False,
                                     shuffle_buffer_size=25000,
                                     labeler=labeler,
                                     seed=0)

    # input_fn = make_input_fn_from_tfrecord(tokenizer=tokenizer, tfrecord=tfrecord_file)

    params = {'batch_size': 4096}

    dataset = input_fn(params)
    sampler = dataset.make_one_shot_iterator()
    sample = sampler.get_next()

    with tf.Session() as sess:
        # confounding = sess.run(sample['confounding'])
        y, y0, y1, t = sess.run([sample['outcome'], sample['y0'], sample['y1'], sample['theorem_referenced']])

    tf.reset_default_graph()

    # print(y0[t==0].mean())
    # print(y1[t==1].mean())

    ground_truth = y1.mean() - y0.mean()
    noiseless_naive = y1[t == 1].mean() - y0[t == 0].mean()
    very_naive = y[t == 1].mean() - y[t == 0].mean()

    return {'ground_truth': ground_truth, 'very_naive': very_naive, 'noiseless_naive': noiseless_naive}


def propensity_sim_ground_truth_and_naive_from_dataset(
        beta0=0.25,
        beta1=0.0,
        gamma=0.0,
        exogenous_confounding=0.0,
        setting='simple',
        tfrecord_file='../dat/PeerRead/proc/arxiv-all.tf_record',
        vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
        base_propensities_path='logs/peerread/arxiv/joint_dragon/seed0/o_accepted_t_buzzy_title/split0/predict/test_results.tsv',
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

    input_fn = make_input_fn_from_file(tfrecord_file,
                                     250,
                                     num_splits=10,
                                     dev_splits=[0],
                                     test_splits=[0],
                                     tokenizer=tokenizer,
                                     is_training=True,
                                     filter_test=False,
                                     shuffle_buffer_size=25000,
                                     labeler=labeler,
                                     seed=0)


    # input_fn = make_input_fn_from_tfrecord(tokenizer=tokenizer, tfrecord=tfrecord_file)

    params = {'batch_size': 4096}

    dataset = input_fn(params)
    sampler = dataset.make_one_shot_iterator()
    sample = sampler.get_next()

    with tf.Session() as sess:
        # confounding = sess.run(sample['confounding'])
        y, y0, y1, t = sess.run([sample['outcome'], sample['y0'], sample['y1'], sample['treatment']])

    tf.reset_default_graph()

    ground_truth = y1.mean() - y0.mean()
    very_naive = y1[t == 1].mean() - y0[t == 0].mean()

    return {'ground_truth': ground_truth, 'noiseless_very_naive': very_naive}


def ate_from_bert_tsv(tsv_path):
    output = pd.read_csv(tsv_path, '\t')
    # output = convert_str_columns_to_float(output)

    y = output['outcome'].values
    t = output['treatment'].values
    q_t0 = output['expected_outcome_st_no_treatment'].values
    q_t1 = output['expected_outcome_st_treatment'].values
    g = output['treatment_probability'].values
    in_test = output['in_test'].values == 1
    in_train = np.logical_not(in_test)

    q_t0_test = q_t0[in_test]
    q_t1_test = q_t1[in_test]
    g_test = g[in_test]
    t_test = t[in_test]
    y_test = y[in_test]

    # all_estimates = ate_estimates(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.03)
    # bonus_estimates = ates_from_atts(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.03)
    # bonuss_estimates = {}
    # for k in bonus_estimates:
    #     bonuss_estimates[k + '_bonus'] = bonus_estimates[k]
    # all_estimates.update(bonuss_estimates)

    # all_estimates = ate_estimates(q_t0, q_t1, g, t, y, truncate_level=0.03)
    all_estimates = ate_estimates(q_t0[in_train], q_t1[in_train], g[in_train], t[in_train], y[in_train],
                                  truncate_level=0.03)

    # print(tsv_path)
    # print(all_estimates)

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
            all_estimates = ate_from_bert_tsv(data_file)
            # print(psi_estimates)
            estimates += [all_estimates]
        except:
            print('wtf')
            print(data_file)

    print(output_dir)

    avg_estimates = {}
    for k in all_estimates.keys():
        k_estimates = []
        for estimate in estimates:
            k_estimates += [estimate[k]]

        avg_estimates[k] = np.mean(k_estimates)
        avg_estimates[(k, 'std')] = np.std(k_estimates)

    return avg_estimates


def process_buzzy_experiment(
        base_dir='logs/peerread/buzzy_based_sim/'):
    # for setting in ['simple', 'multiplicative', 'interaction']:
    outputs = []
    for setting in ['simple']:
        setting_dir = os.path.join(base_dir, "mode" + setting)
        print('---------------------------------')
        print(setting)
        print('---------------------------------')
        for beta1 in [1.0, 5.0, 25.0]:
            print("beta1: {}".format(beta1))
            raw_and_gt = buzzy_sim_ground_truth_and_naive_from_dataset(
                beta0=0.25,
                beta1=beta1,
                gamma=0.0,
                setting=setting,
                tfrecord_file='../dat/PeerRead/proc/arxiv-all.tf_record',
                vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
                seed=0
            )
            output_dir = os.path.join(setting_dir, 'beta00.25.beta1{}.gamma0.0'.format(beta1))
            estimates = bert_psi(output_dir)
            setting_info = {'setting': setting, 'beta1': beta1, 'ground_truth': raw_and_gt['ground_truth']}
            outputs += [{**setting_info, **estimates}]

    return outputs


def process_peerread_application(
        base_dir='logs/peerread/arxiv/joint_dragon/seed0'):
    outputs = []
    for treatment in ['buzzy_title', 'theorem_referenced']:
        treatment_dir = os.path.join(base_dir, "o_accepted_t_" + treatment)
        print('---------------------------------')
        print(treatment)
        print('---------------------------------')
        estimates = bert_psi(treatment_dir)
        setting_info = {'treatment': treatment}
        outputs += [{**setting_info, **estimates}]

    return outputs


def main():
    #
    # gt_and_naive = buzzy_sim_ground_truth_and_naive_from_dataset(
    #     beta0=0.25,
    #     beta1=1.0,
    #     gamma=0.0,
    #     setting='simple',
    #     tfrecord_file='../dat/PeerRead/proc/arxiv-all.tf_record',
    #     vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
    #     seed=0
    # )
    # print(gt_and_naive)

    # gt_and_naive = propensity_sim_ground_truth_and_naive_from_dataset(
    #     beta0=0.25,
    #     beta1=0.0,
    #     gamma=0.0,
    #     exogenous_confounding=0.0,
    #     setting='simple',
    #     tfrecord_file='../dat/PeerRead/proc/arxiv-all.tf_record',
    #     vocab_file="../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt",
    #     base_propensities_path='logs/peerread/arxiv/joint_dragon/seed0/o_accepted_t_buzzy_title/split0/predict/test_results.tsv',
    #     seed=0
    # )
    # print(gt_and_naive)

    print("*****************************************")
    print("Buzzy as Confounding experiment")
    print("*****************************************")

    buzzy_estimates = pd.DataFrame(process_buzzy_experiment())
    print(buzzy_estimates[['beta1', 'ground_truth', 'very_naive', 'q_only', 'tmle', 'bin-tmle']])

    print("*****************************************")
    print("PeerRead application")
    print("*****************************************")

    ate_estimates = pd.DataFrame(process_peerread_application())
    print(ate_estimates.keys())
    print(ate_estimates[['treatment', 'very_naive', 'q_only', 'bin-tmle']])
    print(ate_estimates.loc[:, [('very_naive', 'std'), ('q_only', 'std'),('bin-tmle', 'std')]])


if __name__ == '__main__':
    main()
    # pass
