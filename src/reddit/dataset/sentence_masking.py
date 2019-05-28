# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(token_ids, masked_lm_prob, max_predictions_per_seq, vocab, seed):
    """Creates the predictions for the masked LM objective.

    This should be essentially equivalent to the bits that Bert loads from pre-processed tfrecords

    Except: we just include masks instead of randomly letting the words through or randomly replacing
    """

    basic_mask = tf.less(
        tf.random_uniform(token_ids.shape, minval=0, maxval=1, dtype=tf.float32, seed=seed),
        masked_lm_prob)

    # don't mask special characters or padding
    cand_indexes = tf.logical_and(tf.not_equal(token_ids, vocab["[CLS]"]),
                                  tf.not_equal(token_ids, vocab["[SEP]"]))
    cand_indexes = tf.logical_and(cand_indexes, tf.not_equal(token_ids, 0))
    mask = tf.logical_and(cand_indexes, basic_mask)

    # truncate to max predictions for ease of padding
    masked_lm_positions = tf.where(mask)
    # TODO: it should be essentially impossible for me to see this bug (very unlikely), but I do... symptom of :( ?
    # very rare event: nothing gets picked for mask, causing an irritating bug
    # in this case, just mask the first candidate index
    mlm_shape = tf.shape(masked_lm_positions)[0]
    masked_lm_positions = tf.cond(mlm_shape > 1,
                                  lambda: masked_lm_positions,
                                  lambda: tf.where(cand_indexes)[0:2])

    masked_lm_positions = tf.squeeze(masked_lm_positions)[0:max_predictions_per_seq]
    masked_lm_positions = tf.cast(masked_lm_positions, dtype=tf.int32)
    masked_lm_ids = tf.gather(token_ids, masked_lm_positions)

    mask = tf.cast(
        tf.scatter_nd(tf.expand_dims(masked_lm_positions, 1), tf.ones_like(masked_lm_positions), token_ids.shape),
        bool)

    output_ids = tf.where(mask, vocab["[MASK]"]*tf.ones_like(token_ids), token_ids)

    # pad out to max_predictions_per_seq
    masked_lm_weights = tf.ones_like(masked_lm_ids, dtype=tf.float32) # tracks padding
    add_pad = [[0, max_predictions_per_seq - tf.shape(masked_lm_positions)[0]]]
    masked_lm_weights = tf.pad(masked_lm_weights, add_pad, 'constant')
    masked_lm_positions = tf.pad(masked_lm_positions, add_pad, 'constant')
    masked_lm_ids = tf.pad(masked_lm_ids, add_pad, 'constant')

    return output_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights


def main(_):
    pass


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
