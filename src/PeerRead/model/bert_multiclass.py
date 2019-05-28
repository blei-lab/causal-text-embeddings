"""
Helper to check which categorical attributes of PeerRead are predictable from the text
"""

import tensorflow as tf
import bert.modeling as modeling
import bert.optimization as optimization
from causal_bert.bert_unsupervised import get_masked_lm_output
from causal_bert.logging import make_label_binary_prediction_summaries, binary_label_eval_metric_fn


def _create_unsupervised_only_model(bert, bert_config, features):
    # PeerRead v. reddit inconsistency
    if "op_masked_lm_positions" in features:
        masked_lm_positions = features["op_masked_lm_positions"]
        masked_lm_ids = features["op_masked_lm_ids"]
        masked_lm_weights = features["op_masked_lm_weights"]
    else:
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

    masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs = get_masked_lm_output(
        bert_config, bert.get_sequence_output(), bert.get_embedding_table(),
        masked_lm_positions, masked_lm_ids, masked_lm_weights)
    return masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs


def _make_feedforward_classifier(embedding, labels, num_labels, split, num_hidden_layers, extra_features=None,
                                 label_smoothing=0.01):
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
    if extra_features is None:
        full_embedding = embedding
    else:
        full_embedding = tf.concat([embedding, extra_features], axis=1)

    if num_hidden_layers == 0:
        logits = tf.layers.dense(full_embedding, num_labels, activation=None,
                                 kernel_regularizer=regularizer, bias_regularizer=regularizer)

    else:
        layer = tf.layers.dense(full_embedding, 200, activation=tf.nn.elu)
        for _ in range(num_hidden_layers - 1):
            layer = tf.layers.dense(layer, 200, activation=tf.nn.elu,
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)

        if extra_features is None:
            final_embedding = layer
        else:
            final_embedding = tf.concat([layer, extra_features], axis=1)

        logits = tf.layers.dense(final_embedding, num_labels, activation=None,
                                 kernel_regularizer=regularizer, bias_regularizer=regularizer)

    with tf.name_scope("loss"):
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32,
                                    on_value=1. - label_smoothing, off_value=label_smoothing)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        censored_per_example_loss = split * per_example_loss
        loss = tf.reduce_sum(censored_per_example_loss)

    probabilities = tf.nn.softmax(logits, axis=-1)[:, 1]  # P(T=1)

    return loss, per_example_loss, logits, probabilities


def _get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var  # if ema_var else var

    return ema_getter


def multiclass_model_fn_builder(bert_config, init_checkpoint, learning_rate,
                                num_train_steps, num_warmup_steps, use_tpu,
                                use_one_hot_embeddings, label_pred=True, unsupervised=False,
                                polyak=False, use_extra_features=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        target_name = params['target_name']
        num_labels = params['num_labels']

        labels = features[target_name]

        # because reddit and peerread use slightly different text and pre-training structure
        if "op_token_ids" in features:
            token_mask = features["op_token_mask"]
            maybe_masked_token_ids = features["op_maybe_masked_input_ids"]
        else:
            token_mask = features["token_mask"]
            maybe_masked_token_ids = features["maybe_masked_input_ids"]

        index = features['index']
        in_train = features['in_train']
        in_dev = features['in_dev']
        in_test = features['in_test']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Predictive Model

        bert = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=maybe_masked_token_ids,
            input_mask=token_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs = \
            _create_unsupervised_only_model(bert, bert_config, features)

        bert_embedding = bert.get_pooled_output()

        label_loss, per_example_loss, logits, probabilities = \
            _make_feedforward_classifier(bert_embedding, labels, num_labels, in_train, num_hidden_layers=0,
                                         extra_features=None, label_smoothing=0.01)

        tf.losses.add_loss(masked_lm_loss)
        tf.losses.add_loss(0.1 * label_loss)

        tf.summary.scalar('masked_lm_loss', masked_lm_loss, family='loss')
        tf.summary.scalar('label_loss', label_loss, family='loss')

        total_loss = masked_lm_loss + 0.1 * label_loss

        # some logging
        make_label_binary_prediction_summaries(per_example_loss, logits, labels, in_train, "train")
        make_label_binary_prediction_summaries(per_example_loss, logits, labels, in_dev, "dev")

        # pre-trained model loading
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            # sgd_opt = tf.train.GradientDescentOptimizer(learning_rate)
            # train_op = sgd_opt.minimize(total_loss, global_step=tf.train.get_global_step())

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            pass

        else:
            pass

        return output_spec

    return model_fn
