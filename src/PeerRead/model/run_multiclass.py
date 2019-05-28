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
# See the License for the specific PeerRead governing permissions and
# limitations under the License.

"""BERT finetuning runner
Modified from [TODO: link] from The Google AI Language Team Authors
."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import bert.tokenization as tokenization
import bert.modeling as modeling

from PeerRead.model.bert_multiclass import multiclass_model_fn_builder
from PeerRead.dataset.dataset import make_input_fn_from_file

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "input_files_or_glob", None,
    "The tf_record file (or files) containing the pre-processed data. Probably output of a data_cleaning script.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 10000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "number of warmup steps to take")

# flags.DEFINE_float(
#     "warmup_proportion", 0.1,
#     "Proportion of training to perform linear learning rate warmup for. "
#     "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoints", 1,
                     "How many checkpoints to keep")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer("seed", 0, "Seed for rng.")

flags.DEFINE_bool("label_pred", True, "Whether to do (only) label prediction.")
flags.DEFINE_bool("unsupervised", True, "Whether to do (only) unsupervised training.")

flags.DEFINE_integer("num_splits", 10,
                     "number of splits")
flags.DEFINE_string("dev_splits", '', "indices of development splits")
flags.DEFINE_string("test_splits", '', "indices of test splits")


tf.flags.DEFINE_string(
    "label", "year",
    "categorical label to predict."
)


def main(_):
    print("this program started")
    tf.enable_eager_execution()  # for debugging
    tf.set_random_seed(FLAGS.seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoints,
        # save_checkpoints_steps=None,
        # save_checkpoints_secs=None,
        save_summary_steps=10,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # Estimator and data pipeline setup

    label_dict = \
        {'accepted': 2,
         'abstract_contains_deep': 2,
         'abstract_contains_neural': 2,
         'abstract_contains_embedding': 2,
         'abstract_contains_outperform': 2,
         'abstract_contains_novel': 2,
         'abstract_contains_state_of_the_art': 2,
         'abstract_contains_state-of-the-art': 2,
         'title_contains_deep': 2,
         'title_contains_neural': 2,
         'title_contains_embedding': 2,
         'contains_appendix': 2,
         'theorem_referenced': 2,
         'equation_referenced': 2,
         'year': 11,
         'venue': 9,
         'arxiv': 3}

    params = {'target_name': FLAGS.label, 'num_labels': label_dict[FLAGS.label]}
    dev_splits = [int(s) for s in str.split(FLAGS.dev_splits)]
    test_splits = [int(s) for s in str.split(FLAGS.test_splits)]

    num_train_steps = FLAGS.num_train_steps
    num_warmup_steps = FLAGS.num_warmup_steps

    model_fn = multiclass_model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        label_pred=FLAGS.label_pred,
        unsupervised=FLAGS.unsupervised,
        polyak=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        params=params)

    if FLAGS.do_train:
        input_files_or_glob = FLAGS.input_files_or_glob

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        # subsample and process the data
        with tf.name_scope("training_data"):
            train_input_fn = make_input_fn_from_file(
                input_files_or_glob=input_files_or_glob,
                seq_length=FLAGS.max_seq_length,
                num_splits=FLAGS.num_splits,
                dev_splits=dev_splits,
                test_splits=test_splits,
                tokenizer=tokenizer,
                is_training=True,
                shuffle_buffer_size=25000,  # note: bert hardcoded this, and I'm following suit
                seed=FLAGS.seed)

        # additional logging
        hooks = []
        if FLAGS.label_pred:
            hooks += [
                tf.train.LoggingTensorHook({
                    # 'token_ids': 'token_ids',
                    # 'token_mask': 'token_mask',
                    # 'label_ids': 'label_ids',
                    # 'pred_in': 'summary/in_split/predictions',
                    # 'pred_out': 'summary/out_split/predictions',
                    # 'ra_in': 'summary/in_split/labels/kappa/batch_random_agreement/random_agreement',
                    # 'ra_out': 'summary/out_split/labels/kappa/batch_random_agreement/random_agreement',
                },
                    every_n_iter=1000)
            ]

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=hooks)

    if FLAGS.do_train and (FLAGS.do_eval or FLAGS.do_predict):
        # reload the model to get rid of dropout and input token masking
        trained_model_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
        model_fn = multiclass_model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=trained_model_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu,
            label_pred=True,
            unsupervised=False,
            polyak=False
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            params=params)

    if FLAGS.do_eval:

        tf.logging.info("***** Running evaluation *****")
        # tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            pass
            # eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = make_input_fn_from_file(
                input_files_or_glob=FLAGS.input_files_or_glob,
                seq_length=FLAGS.max_seq_length,
                num_splits=FLAGS.num_splits,
                dev_splits=dev_splits,
                test_splits=test_splits,
                tokenizer=tokenizer,
                is_training=False,
                filter_test=False,
                seed=FLAGS.seed)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        tf.logging.info("***** Running prediction*****")

        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_input_fn = make_input_fn_from_file(
                input_files_or_glob=FLAGS.input_files_or_glob,
                seq_length=FLAGS.max_seq_length,
                num_splits=FLAGS.num_splits,
                dev_splits=dev_splits,
                test_splits=test_splits,
                tokenizer=tokenizer,
                is_training=False,
                filter_test=False,
                seed=FLAGS.seed)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")

            attribute_names = ['in_test',
                               'treatment_probability',
                               'expected_outcome_st_treatment', 'expected_outcome_st_no_treatment',
                               'outcome', 'treatment']

            header = "\t".join(
                attribute_name for attribute_name in attribute_names) + "\n"
            writer.write(header)
            for prediction in result:
                output_line = "\t".join(
                    str(prediction[attribute_name]) for attribute_name in attribute_names) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_files_or_glob")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
