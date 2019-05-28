"""

Parses PeerRead data into a Bert-based model compatible format, and stores as tfrecord

See dataset.py for the corresponding code to read this data

"""
import argparse
import numpy as np
import pandas as pd
from scipy.special import logit, expit

import tensorflow as tf
try:
    import mkl_random as random
except ImportError:
    import numpy.random as random

import bert.tokenization as tokenization
from PeerRead.dataset.sentence_masking import create_masked_lm_predictions

# hardcoded because protobuff is not self describing for some bizarre reason
all_context_features = \
    {'accepted': tf.int64,
     'most_recent_reference_year': tf.int64,
     'num_recent_references': tf.int64,
     'num_references': tf.int64,
     'num_refmentions': tf.int64,
     # 'avg_length_reference_mention_contexts': tf.float32,
     'abstract_contains_deep': tf.int64,
     'abstract_contains_neural': tf.int64,
     'abstract_contains_embedding': tf.int64,
     'abstract_contains_outperform': tf.int64,
     'abstract_contains_novel': tf.int64,
     'abstract_contains_state-of-the-art': tf.int64,
     "title_contains_deep": tf.int64,
     "title_contains_neural": tf.int64,
     "title_contains_embedding": tf.int64,
     "title_contains_gan": tf.int64,
     'num_ref_to_figures': tf.int64,
     'num_ref_to_tables': tf.int64,
     'num_ref_to_sections': tf.int64,
     'num_uniq_words': tf.int64,
     'num_sections': tf.int64,
     # 'avg_sentence_length': tf.float32,
     'contains_appendix': tf.int64,
     'title_length': tf.int64,
     'num_authors': tf.int64,
     'num_ref_to_equations': tf.int64,
     'num_ref_to_theorems': tf.int64,
     'id': tf.int64,
     'year': tf.int64,
     'venue': tf.int64,
     'arxiv': tf.int64,
     'many_split': tf.int64}


def compose(*fns):
    """ Composes the given functions in reverse order.

    Parameters
    ----------
    fns: the functions to compose

    Returns
    -------
    comp: a function that represents the composition of the given functions.
    """
    import functools

    def _apply(x, f):
        if isinstance(x, tuple):
            return f(*x)
        else:
            return f(x)

    def comp(*args):
        return functools.reduce(_apply, fns, args)

    return comp


def make_parser(abs_seq_len=250):
    context_features = {k: tf.FixedLenFeature([], dtype=v) for k, v in all_context_features.items()}

    abstract_features = {
        "token_ids": tf.FixedLenFeature([abs_seq_len], tf.int64),
        "token_mask": tf.FixedLenFeature([abs_seq_len], tf.int64),
        # "segment_ids": tf.FixedLenFeature([abs_seq_len], tf.int64),
    }

    _name_to_features = {**context_features, **abstract_features}

    def parser(record):
        tf_example = tf.parse_single_example(
            record,
            features=_name_to_features
        )

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(tf_example.keys()):
            t = tf_example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            tf_example[name] = t

        return tf_example

    return parser


def make_input_id_masker(tokenizer, seed):
    # (One of) Bert's unsupervised objectives is to mask some fraction of the input words and predict the masked words

    def masker(data):
        token_ids = data['token_ids']
        maybe_masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = create_masked_lm_predictions(
            token_ids,
            # pre-training defaults from Bert docs
            masked_lm_prob=0.15,
            max_predictions_per_seq=20,
            vocab=tokenizer.vocab,
            seed=seed)
        return {
            **data,
            'maybe_masked_input_ids': maybe_masked_input_ids,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_ids': masked_lm_ids,
            'masked_lm_weights': masked_lm_weights
        }

    return masker


def make_extra_feature_cleaning():
    def extra_feature_cleaning(data):
        data['num_authors'] = tf.minimum(data['num_authors'], 6)-1
        data['year'] = data['year']-2007

        # some extras
        equation_referenced = tf.minimum(data['num_ref_to_equations'], 1)
        theorem_referenced = tf.minimum(data['num_ref_to_theorems'], 1)

        # buzzy title
        any_buzz = data["title_contains_deep"] + data["title_contains_neural"] + \
                   data["title_contains_embedding"] + data["title_contains_gan"]
        buzzy_title = tf.cast(tf.not_equal(any_buzz, 0), tf.int32)

        return {**data,
                'equation_referenced': equation_referenced,
                'theorem_referenced': theorem_referenced,
                'buzzy_title': buzzy_title,
                'index': data['id']}
    return extra_feature_cleaning


def make_label():
    """
    Do something slightly nuts for testing purposes
    :return:
    """
    def labeler(data):
        return {**data, 'label_ids': data['accepted']}

    # def wacky_labeler(data):
    #     label_ids = tf.greater_equal(data['num_authors'], 4)
    #     label_ids = tf.cast(label_ids, tf.int32)
    #     return {**data, 'label_ids': label_ids}

    return labeler


def outcome_sim(beta0, beta1, gamma, treatment, confounding, noise, setting="simple"):
    if setting == "simple":
        y0 = beta1 * confounding
        y1 = beta0 + y0

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    elif setting == "multiplicative":
        y0 = beta1 * confounding
        y1 = beta0 * y0

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    elif setting == "interaction":
        # required to distinguish ATT and ATE
        y0 = beta1 * confounding
        y1 = y0 + beta0 * tf.math.square(confounding)

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    else:
        raise Exception('setting argument to make_simulated_labeler not recognized')

    return simulated_score, y0, y1


def _make_hidden_float_constant(value, name):
    # hack to prevent tensorflow from writing the constant to the graphdef
    return tf.py_func(
        lambda: value,
        [], tf.float32, stateful=False,
        name=name)


def make_buzzy_based_simulated_labeler(treat_strength, con_strength, noise_level, setting="simple", seed=0):
    # hardcode probability of theorem given buzzy / not_buzzy
    theorem_given_buzzy_probs = np.array([0.27, 0.07], dtype=np.float32)

    np.random.seed(seed)
    all_noise = np.array(random.normal(0, 1, 12000), dtype=np.float32)
    all_threshholds = np.array(random.uniform(0, 1, 12000), dtype=np.float32)

    def labeler(data):
        buzzy = data['buzzy_title']
        index = data['index']
        treatment = data['theorem_referenced']
        treatment = tf.cast(treatment, tf.float32)
        confounding = 3.0*(tf.gather(theorem_given_buzzy_probs, buzzy) - 0.25)

        noise = tf.gather(all_noise, index)

        y, y0, y1 = outcome_sim(treat_strength, con_strength, noise_level, treatment, confounding, noise, setting=setting)
        simulated_prob = tf.nn.sigmoid(y)
        y0 = tf.nn.sigmoid(y0)
        y1 = tf.nn.sigmoid(y1)
        threshold = tf.gather(all_threshholds, index)
        simulated_outcome = tf.cast(tf.greater(simulated_prob, threshold), tf.int32)

        return {**data, 'outcome': simulated_outcome, 'y0': y0, 'y1': y1}

    return labeler


def make_propensity_based_simulated_labeler(treat_strength, con_strength, noise_level,
                                            base_propensity_scores, example_indices, exogeneous_con=0.,
                                            setting="simple", seed=42):
    np.random.seed(seed)
    all_noise = random.normal(0, 1, base_propensity_scores.shape[0]).astype(np.float32)
    all_threshholds = np.array(random.uniform(0, 1, base_propensity_scores.shape[0]), dtype=np.float32)

    extra_confounding = random.normal(0, 1, base_propensity_scores.shape[0]).astype(np.float32)

    all_propensity_scores = expit((1.-exogeneous_con)*logit(base_propensity_scores) + exogeneous_con * extra_confounding).astype(np.float32)
    all_treatments = random.binomial(1, all_propensity_scores).astype(np.int32)

    # indices in dataset refer to locations in entire corpus,
    # but propensity scores will typically only inlcude a subset of the examples
    reindex_hack = np.zeros(12000, dtype=np.int32)
    reindex_hack[example_indices] = np.arange(example_indices.shape[0], dtype=np.int32)

    def labeler(data):
        index = data['index']
        index_hack = tf.gather(reindex_hack, index)
        treatment = tf.gather(all_treatments, index_hack)
        confounding = 3.0 * (tf.gather(all_propensity_scores, index_hack) - 0.25)
        noise = tf.gather(all_noise, index_hack)

        y, y0, y1 = outcome_sim(treat_strength, con_strength, noise_level, tf.cast(treatment, tf.float32), confounding, noise, setting=setting)
        simulated_prob = tf.nn.sigmoid(y)
        y0 = tf.nn.sigmoid(y0)
        y1 = tf.nn.sigmoid(y1)
        threshold = tf.gather(all_threshholds, index)
        simulated_outcome = tf.cast(tf.greater(simulated_prob, threshold), tf.int32)

        return {**data, 'outcome': simulated_outcome, 'y0': y0, 'y1': y1, 'treatment': treatment}

    return labeler


def make_split_document_labels(num_splits, dev_splits, test_splits):
    """
    Adapts tensorflow dataset to produce additional elements that indicate whether each datapoint is in train, dev,
    or test

    Particularly, splits the data into num_split folds, and censors the censored_split fold

    Parameters
    ----------
    num_splits integer in [0,100)
    dev_splits list of integers in [0,num_splits)
    test_splits list of integers in [0, num_splits)

    Returns
    -------
    fn: A function that can be used to map a dataset to censor some of the document labels.
    """
    def _tf_in1d(a,b):
        """
        Tensorflow equivalent of np.in1d(a,b)
        """
        a = tf.expand_dims(a, 0)
        b = tf.expand_dims(b, 1)
        return tf.reduce_any(tf.equal(a, b), 1)

    def _tf_scalar_a_in1d_b(a, b):
        """
        Tensorflow equivalent of np.in1d(a,b)
        """
        return tf.reduce_any(tf.equal(a, b))

    def fn(data):
        many_split = data['many_split']
        reduced_split = tf.floormod(many_split, num_splits)  # reduce the many splits to just num_splits

        in_dev = _tf_scalar_a_in1d_b(reduced_split, dev_splits)
        in_test = _tf_scalar_a_in1d_b(reduced_split, test_splits)
        in_train = tf.logical_not(tf.logical_or(in_dev, in_test))

        # in_dev = _tf_in1d(reduced_splits, dev_splits)
        # in_test = _tf_in1d(reduced_splits, test_splits)
        # in_train = tf.logical_not(tf.logical_or(in_dev, in_test))

        # code expects floats
        in_dev = tf.cast(in_dev, tf.float32)
        in_test = tf.cast(in_test, tf.float32)
        in_train = tf.cast(in_train, tf.float32)

        return {**data, 'in_dev': in_dev, 'in_test': in_test, 'in_train': in_train}

    return fn


def dataset_processing(dataset, parser, masker, labeler, is_training, num_splits, dev_splits, test_splits, batch_size,
                       filter_test=False,
                       shuffle_buffer_size=100):
    """

    Parameters
    ----------
    dataset  tf.data dataset
    parser function, read the examples, should be based on tf.parse_single_example
    masker function, should provide Bert style masking
    labeler function, produces labels
    is_training
    num_splits
    censored_split
    batch_size
    filter_test restricts to only examples where in_test=1
    shuffle_buffer_size

    Returns
    -------

    """

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    data_processing = compose(parser,  # parse from tf_record
                              labeler,  # add a label (unused downstream at time of comment)
                              make_split_document_labels(num_splits, dev_splits, test_splits),  # censor some labels
                              masker)  # Bert style token masking for unsupervised training

    dataset = dataset.map(data_processing, 4)

    if filter_test:
        def filter_test_fn(data):
            return tf.equal(data['in_test'], 1)

        dataset = dataset.filter(filter_test_fn)

    if is_training:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

    return dataset


def make_input_fn_from_file(input_files_or_glob, seq_length,
                            num_splits, dev_splits, test_splits,
                            tokenizer, is_training,
                            filter_test=False,
                            shuffle_buffer_size=100, seed=0, labeler=None):

    input_files = []
    for input_pattern in input_files_or_glob.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    if labeler is None:
        labeler = make_label()

    def input_fn(params):
        batch_size = params["batch_size"]

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=len(input_files))
            cycle_length = min(4, len(input_files))

        else:
            dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            cycle_length = 1  # go through the datasets in a deterministic order

        # make the record parsing ops
        max_abstract_len = seq_length

        parser = make_parser(max_abstract_len)  # parse the tf_record
        parser = compose(parser, make_extra_feature_cleaning())
        masker = make_input_id_masker(tokenizer, seed)  # produce masked subsets for unsupervised training

        # for use with interleave
        def _dataset_processing(input):
            input_dataset = tf.data.TFRecordDataset(input)
            processed_dataset = dataset_processing(input_dataset,
                                                   parser, masker, labeler,
                                                   is_training,
                                                   num_splits, dev_splits, test_splits,
                                                   batch_size, filter_test, shuffle_buffer_size)
            return processed_dataset

        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                _dataset_processing,
                sloppy=is_training,
                cycle_length=cycle_length))

        return dataset

    return input_fn


def main():
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_buffer_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_abs_len', type=int, default=250)

    args = parser.parse_args()

    # for easy debugging
    # filename = "../../dat/PeerRead/proc/acl_2017.tf_record"
    # filename = glob.glob('../dat/PeerRead/proc/*.tf_record')
    filename = '../dat/PeerRead/proc/arxiv-all.tf_record'

    vocab_file = "../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt"
    seed = 0
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    num_splits = 10
    # dev_splits = [0]
    # test_splits = [0]
    dev_splits = []
    test_splits = [1,2]


    labeler = make_buzzy_based_simulated_labeler(0.5, 5.0, 0.0, 'simple',
                                                 seed=0)

    base_propensities_path = 'logs/peerread/buzzy_based_sim/modesimple/beta00.25.beta11.0.gamma0.0/split0/predict/test_results.tsv'
    output = pd.read_csv(base_propensities_path, '\t')
    base_propensity_scores = np.concatenate([output['treatment_probability'].values, np.zeros(8)])
    example_indices = output['index'].values

    labeler = make_propensity_based_simulated_labeler(treat_strength=0.25,
                                                      con_strength=1.0,
                                                      noise_level=0.0,
                                                      base_propensity_scores=base_propensity_scores,
                                                      example_indices=example_indices,
                                                      exogeneous_con=0.,
                                                      setting="simple",
                                                      seed=0)
    # labeler = None

    input_dataset_from_filenames = make_input_fn_from_file(filename,
                                                           250,
                                                           num_splits,
                                                           dev_splits,
                                                           test_splits,
                                                           tokenizer,
                                                           is_training=True,
                                                           filter_test=False,
                                                           shuffle_buffer_size=25000,
                                                           labeler=labeler,
                                                           seed=0)
    params = {'batch_size': 4096}
    input_dataset = input_dataset_from_filenames(params)

    # masker = make_input_id_masker(tokenizer, seed)
    # parser = make_parser(args.max_abs_len)
    # labeler = make_label()
    #
    # dataset = tf.data.TFRecordDataset(filename)

    # input_dataset = dataset_processing(dataset, parser, masker, labeler,
    #                                    is_training=True, num_examples=num_examples, split_indices=split_indices,
    #                                    batch_size=args.batch_size, shuffle_buffer_size=100)

    secs = []

    itr = input_dataset.make_one_shot_iterator()
    # print(itr.get_next()["token_ids"].name)
    # for i in range(1000):
    #     sample = itr.get_next()

    for i in range(10):
        sample = itr.get_next()
        print(np.max(sample['index']))
        print(np.min(sample['index']))
        # "title_contains_deep": tf.int64,
        # "title_contains_neural": tf.int64,
        # "title_contains_embedding": tf.int64,
        # "title_contains_gan": tf.int64,
        # print(sample['buzzy_title'])
        # venue = sample['venue']
        # arxiv = sample['arxiv']
        # print("venue: {}".format(venue))
        # print("arxiv: {}".format(arxiv))
        #
        # print("frac_arxiv: {}".format((np.equal(arxiv,0)).mean()))

        # print("year: {}".format(sample['year']))

        # thm_ref = sample['theorem_referenced'].numpy()
        # buzzy_title = sample['buzzy_title'].numpy()
        #
        # t_pr = thm_ref.mean()
        # b_pr = buzzy_title.mean()
        #
        # tb_pr = (thm_ref*buzzy_title).mean()
        # tnotb_pr = (thm_ref*(1.-buzzy_title)).mean()

        # print("t_pr: {}".format(t_pr))
        # print("b_pr: {}".format(b_pr))
        # print("th given buzzy: {}".format(tb_pr / b_pr))
        # print("th given not buzzy: {}".format(tnotb_pr / (1-b_pr)))

        # treatment = sample['theorem_referenced'].numpy()
        # treatment = sample['treatment']
        # print("treatment: {}".format(treatment.mean()))
        # outcome = sample['outcome'].numpy()
        # print("outcome: {}".format(outcome.mean()))
        #
        # print("outcome_st_treatment: {}".format((outcome*treatment).mean()/treatment.mean()))
        # print("outcome_st_not_treatment: {}".format((outcome*(1.-treatment)).mean()/(1.-treatment).mean()))
        #
        # # print("outcome: {}".format(tf.reduce_mean(tf.cast(sample['outcome'], tf.float32))))
        # print("y0: {}".format(sample['y0']))
        # print("y1: {}".format(sample['y1']))

if __name__ == "__main__":
    main()