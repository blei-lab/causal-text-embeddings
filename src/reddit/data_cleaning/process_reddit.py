"""
Pre-process raw reddit data into tfrecord.
"""

import argparse
import os
import random

import tensorflow as tf
import numpy as np
import bert.tokenization as tokenization
import reddit.data_cleaning.reddit_posts as rp

rng = random.Random(0)


def process_without_response_task(row_dict, tokenizer):
    context_features = {}
    op_tokens = tokenizer.tokenize(row_dict['post_text'])

    text_features = {'op_text': op_tokens}
    for key in row_dict:
        if key not in {'post_text', 'response_text'}:
            context_features[key] = row_dict[key]

    return text_features, context_features


def process_row_record(row_dict, tokenizer, random_response=None, use_response_task=True):
    if not use_response_task:
        return process_without_response_task(row_dict, tokenizer)

    context_features = {}
    op_tokens = tokenizer.tokenize(row_dict['post_text'])

    if random_response:
        response_tokens = tokenizer.tokenize(random_response)
        context_features['has_random_resp'] = 1
    else:
        response_tokens = tokenizer.tokenize(row_dict['response_text'])
        context_features['has_random_resp'] = 0

    if len(op_tokens) < 2 or len(response_tokens) < 2:
        return None, None

    text_features = {'op_text': op_tokens,
                     'resp_text': response_tokens}

    for key in row_dict:
        if key not in {'post_text', 'response_text'}:
            context_features[key] = row_dict[key]

    # add hand crafted features from PeerRead

    return text_features, context_features


def bert_process_sentence(example_tokens, max_seq_length, tokenizer, segment=1):
    """
    Tokenization and pre-processing of text as expected by Bert

    Parameters
    ----------
    example_tokens
    max_seq_length
    tokenizer

    Returns
    -------



    """
    # Account for [CLS] and [SEP] with "- 2"
    if len(example_tokens) > max_seq_length - 2:
        example_tokens = example_tokens[0:(max_seq_length - 2)]

    # The convention in BERT for single sequences is:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence.

    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    '''
    If we need sentence detection logic:

    if token in string.punctuation:
        #     if (tidx < len(example_tokens) - 1) and (example_tokens[tidx + 1] in string.punctuation):
        #         tokens.append(token)
        #     else:
        #         tokens.append("[SEP]")
        # else:

    '''

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(segment)
    for tidx, token in enumerate(example_tokens):
        tokens.append(token)
        segment_ids.append(segment)

    tokens.append("[SEP]")
    segment_ids.append(segment)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def reddit_to_bert_Example(text_features, context_features, max_seq_length, tokenizer, use_response_task=True):
    """
    Parses the input paper into a tf.Example as expected by Bert
    Note: the docs for tensorflow Example are awful ¯\_(ツ)_/¯
    """
    features = {}

    op_tokens, op_padding_mask, op_segments = \
        bert_process_sentence(text_features['op_text'], max_seq_length, tokenizer)

    features["op_token_ids"] = _int64_feature(op_tokens)
    features["op_token_mask"] = _int64_feature(op_padding_mask)
    features["op_segment_ids"] = _int64_feature(op_segments)

    if use_response_task:
        resp_tokens, resp_padding_mask, resp_segments = \
            bert_process_sentence(text_features['resp_text'], max_seq_length, tokenizer, segment=0)

        features["resp_token_ids"] = _int64_feature(resp_tokens)
        features["resp_token_mask"] = _int64_feature(resp_padding_mask)
        features["resp_segment_ids"] = _int64_feature(resp_segments)

    # abstract_features["segment_ids"] = create_int_feature(feature.segment_ids)  TODO: ommission may cause bugs
    # abstract_features["label_ids"] = _int64_feature([feature.label_id])

    # non-sequential features
    tf_context_features, tf_context_features_types = _dict_of_nonlist_numerical_to_tf_features(context_features)

    features = {**tf_context_features, **features}

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    return tf_example


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto,
    e.g, An integer label.
    """
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Wrapper for inserting a float Feature into a SequenceExample proto,
    e.g, An integer label.
    """
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto,
    e.g, an image in byte
    """
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _dict_of_nonlist_numerical_to_tf_features(my_dict):
    """
    Strip out non-numerical features
    Returns tf_features_dict: a dictionary suitable for passing to tf.train.example
            tf_types_dict: a dictionary of the tf types of previous dict

    """

    tf_types_dict = {}
    tf_features_dict = {}
    for k, v in my_dict.items():
        if isinstance(v, int) or isinstance(v, bool):
            tf_features_dict[k] = _int64_feature(v)
            tf_types_dict[k] = tf.int64
        elif isinstance(v, float):
            tf_features_dict[k] = _float_feature(v)
            tf_types_dict[k] = tf.float32
        else:
            pass

    return tf_features_dict, tf_types_dict


def process_reddit_dataset(data_dir, out_dir, out_file, max_abs_len, tokenizer, subsample, use_latest_reddit):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if use_latest_reddit:
        if data_dir:
            reddit_df = rp.load_reddit(path=data_dir, use_latest=use_latest_reddit, convert_columns=True)
        else:
            reddit_df = rp.load_reddit(use_latest=use_latest_reddit, convert_columns=True)

    else:
        if data_dir:
            reddit_df = rp.load_reddit(path=data_dir, convert_columns=True)
        else:
            reddit_df = rp.load_reddit(convert_columns=True)

    # add persistent record of the index of the data examples
    reddit_df['index'] = reddit_df.index

    reddit_records = reddit_df.to_dict('records')
    random_example_indices = np.arange(len(reddit_records))
    np.random.shuffle(random_example_indices)
    random_response_mask = np.random.randint(0, 2, len(reddit_records))

    with tf.python_io.TFRecordWriter(out_dir + "/" + out_file) as writer:
        for idx, row_dict in enumerate(reddit_records):

            if subsample and idx >= subsample:
                break

            if (random_response_mask[idx]) and (random_example_indices[idx] != idx):
                random_response = reddit_records[random_example_indices[idx]]['response_text']
                text_features, context_features = process_row_record(row_dict, tokenizer,
                                                                     random_response=random_response)
            else:
                text_features, context_features = process_row_record(row_dict, tokenizer)
            '''
            TODO: is this needed?

            many_split = rng.randint(0, 100)  # useful for easy data splitting later
            extra_context = {'id': idx, 'many_split': many_split}
            context_features.update(extra_context)
            '''

            # turn it into a tf.data example

            if text_features and context_features:
                many_split = rng.randint(0, 100)  # useful for easy data splitting later
                extra_context = {'many_split': many_split}
                context_features.update(extra_context)

                row_ex = reddit_to_bert_Example(text_features, context_features,
                                                max_seq_length=max_abs_len,
                                                tokenizer=tokenizer)
                writer.write(row_ex.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='../dat/reddit')
    parser.add_argument('--out-file', type=str, default='proc.tf_record')
    parser.add_argument('--vocab-file', type=str, default='../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt')
    parser.add_argument('--max-abs-len', type=int, default=128)
    parser.add_argument('--subsample', type=int, default=0)
    parser.add_argument('--use-latest-reddit', type=bool, default=True)

    args = parser.parse_args()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=True)

    process_reddit_dataset(args.data_dir, args.out_dir, args.out_file,
                           args.max_abs_len, tokenizer, args.subsample, args.use_latest_reddit)


if __name__ == "__main__":
    main()
