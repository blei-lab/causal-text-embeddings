"""
Simple pre-processing for PeerRead papers.
Takes in JSON formatted data from ScienceParse and outputs a tfrecord


Reference example:
https://github.com/tensorlayer/tensorlayer/blob/9528da50dfcaf9f0f81fba9453e488a1e6c8ee8f/examples/data_process/tutorial_tfrecord3.py
"""

import argparse
import glob
import os
import random

import io
import json
from dateutil.parser import parse as parse_date

import tensorflow as tf

import bert.tokenization as tokenization
from PeerRead.ScienceParse.Paper import Paper
from PeerRead.ScienceParse.ScienceParseReader import ScienceParseReader
from PeerRead.data_cleaning.PeerRead_hand_features import get_PeerRead_hand_features

rng = random.Random(0)


def process_json_paper(paper_json_filename, scienceparse_dir, tokenizer):
    paper = Paper.from_json(paper_json_filename)
    paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT,
                                                               scienceparse_dir)

    # tokenize PeerRead features
    try:
        title_tokens = tokenizer.tokenize(paper.TITLE)
    except ValueError:  # missing titles are quite common sciparse
        print("Missing title for " + paper_json_filename)
        title_tokens = None

    abstract_tokens = tokenizer.tokenize(paper.ABSTRACT)

    text_features = {'title': title_tokens,
                     'abstract': abstract_tokens}

    context_features = {'authors': paper.AUTHORS,
                        'accepted': paper.ACCEPTED,
                        'name': paper.ID}

    # add hand crafted features from PeerRead
    pr_hand_features = get_PeerRead_hand_features(paper)
    context_features.update(pr_hand_features)

    return text_features, context_features


def bert_process_sentence(example_tokens, max_seq_length, tokenizer):
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
    # sequence or the second sequence. (vv: Not relevant for us)

    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    # vv: segment_ids seem to be the same as type_ids
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in example_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)


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


def paper_to_bert_Example(text_features, context_features, max_seq_length, tokenizer):
    """
    Parses the input paper into a tf.Example as expected by Bert
    Note: the docs for tensorflow Example are awful ¯\_(ツ)_/¯
    """
    abstract_features = {}

    abstract_tokens, abstract_padding_mask, _ = \
        bert_process_sentence(text_features['abstract'], max_seq_length, tokenizer)

    abstract_features["token_ids"] = _int64_feature(abstract_tokens)
    abstract_features["token_mask"] = _int64_feature(abstract_padding_mask)
    # abstract_features["segment_ids"] = create_int_feature(feature.segment_ids)  TODO: ommission may cause bugs
    # abstract_features["label_ids"] = _int64_feature([feature.label_id])

    # non-sequential features
    tf_context_features, tf_context_features_types = _dict_of_nonlist_numerical_to_tf_features(context_features)

    features = {**tf_context_features, **abstract_features}

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


venues = {'acl': 1,
          'conll': 2,
          'iclr': 3,
          'nips': 4,
          'icml': 5,
          'emnlp': 6,
          'aaai': 7,
          'hlt-naacl': 8,
          'arxiv': 0}


def _venues(venue_name):
    if venue_name.lower() in venues:
        return venues[venue_name.lower()]
    else:
        return -1


def _arxiv_subject(subjects):
    subject = subjects[0]
    if 'lg' in subject.lower():
        return 0
    elif 'cl' in subject.lower():
        return 1
    elif 'ai' in subject.lower():
        return 2
    else:
        raise Exception("arxiv subject not recognized")


def clean_PeerRead_dataset(review_json_dir, parsedpdf_json_dir,
                           venue, year,
                           out_dir, out_file,
                           max_abs_len, tokenizer,
                           default_accept=1,
                           is_arxiv = False):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('Reading reviews from...', review_json_dir)
    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_json_dir)))

    with tf.python_io.TFRecordWriter(out_dir + "/" + out_file) as writer:
        for idx, paper_json_filename in enumerate(paper_json_filenames):
            text_features, context_features = process_json_paper(paper_json_filename, parsedpdf_json_dir, tokenizer)

            if context_features['accepted'] is None:  # missing for conferences other than ICLR (we only see accepts)
                context_features['accepted'] = default_accept

            many_split = rng.randint(0, 100)  # useful for easy data splitting later

            # other context features
            arxiv = -1
            if is_arxiv:
                with io.open(paper_json_filename) as json_file:
                    loaded = json.load(json_file)
                year = parse_date(loaded['DATE_OF_SUBMISSION']).year
                venue = _venues(loaded['conference'])
                arxiv = _arxiv_subject([loaded['SUBJECTS']])

            extra_context = {'id': idx, 'venue': venue, 'year': year, 'many_split': many_split,
                             'arxiv': arxiv}
            context_features.update(extra_context)

            # turn it into a tf.data example
            paper_ex = paper_to_bert_Example(text_features, context_features,
                                             max_seq_length=max_abs_len, tokenizer=tokenizer)
            writer.write(paper_ex.SerializeToString())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--review-json-dir', type=str, default='../dat/PeerRead/arxiv.all/all/reviews')
    parser.add_argument('--parsedpdf-json-dir', type=str, default='../dat/PeerRead/arxiv.all/all/parsed_pdfs')
    parser.add_argument('--out-dir', type=str, default='../dat/PeerRead/proc')
    parser.add_argument('--out-file', type=str, default='arxiv-all.tf_record')
    parser.add_argument('--vocab-file', type=str, default='../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt')
    parser.add_argument('--max-abs-len', type=int, default=250)
    parser.add_argument('--venue', type=int, default=0)
    parser.add_argument('--year', type=int, default=2017)


    args = parser.parse_args()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=True)

    clean_PeerRead_dataset(args.review_json_dir, args.parsedpdf_json_dir,
                           args.venue, args.year,
                           args.out_dir, args.out_file,
                           args.max_abs_len, tokenizer, is_arxiv=True)


if __name__ == "__main__":
    main()
