import tensorflow as tf
import bert.modeling as modeling
import bert.optimization as optimization
from causal_bert.bert_unsupervised import get_masked_lm_output
from causal_bert.logging import make_label_regression_prediction_summaries, make_label_binary_prediction_summaries, \
    binary_label_eval_metric_fn, cont_label_eval_metric_fn


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


def _make_continuous_feedforward_regressor(embedding, outcomes, split, num_hidden_layers, extra_features=None):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    if extra_features is None:
        full_embedding = embedding
    else:
        full_embedding = tf.concat([embedding, extra_features], axis=1)

    if num_hidden_layers == 0:
        prediction = tf.layers.dense(full_embedding, 1, activation=None,
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

        prediction = tf.layers.dense(final_embedding, 1, activation=None,
                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)

    with tf.name_scope("loss"):
        labels_float = tf.dtypes.cast(outcomes, tf.float32)
        labels_float = tf.expand_dims(labels_float, 1)

        per_example_loss = tf.reduce_sum(tf.square(labels_float - prediction), axis=-1)
        censored_per_example_loss = split * per_example_loss
        loss = tf.reduce_sum(censored_per_example_loss)

        # TODO: debugging code
        labels_float = tf.identity(labels_float, 'labels_float')
        prediction = tf.identity(prediction, 'prediction')
        per_example_loss = tf.identity(per_example_loss, 'per_example_loss')

    # probabilities = tf.nn.softmax(logits, axis=-1)[:, 1]  # P(T=1)

    return loss, per_example_loss, prediction


def _create_or_get_dragonnet_cont_outcome(embedding, treatment, outcome, split, extra_features=None, getter=None):
    """
    Make predictions for the outcome, using the treatment and embedding,
    and predictions for the treatment, using the embedding
    The outcome is assumed to be continuous, the treatment binary

    Note that we return the loss as a sum (and not a mean). This makes more sense for training dynamics

    Parameters
    ----------
    bert
    is_training
    treatment
    outcome
    label_dict
    split
    getter custom getter, for polyak averaging support

    Returns
    -------

    """

    treatment_float = tf.cast(treatment, tf.float32)
    with tf.variable_scope('dragon_net', reuse=tf.AUTO_REUSE, custom_getter=getter):
        with tf.variable_scope('treatment'):
            loss_t, per_example_loss_t, logits_t, expectation_t = _make_feedforward_classifier(
                embedding, treatment, 2, split, num_hidden_layers=0, extra_features=extra_features)

        with tf.variable_scope('outcome_st_treatment'):
            loss_ot1, per_example_loss_ot1, expectation_ot1 = _make_continuous_feedforward_regressor(
                embedding, outcome, split=split * treatment_float, num_hidden_layers=2,
                extra_features=extra_features)

        with tf.variable_scope('outcome_st_no_treatment'):
            loss_ot0, per_example_loss_ot0, expectation_ot0 = _make_continuous_feedforward_regressor(
                embedding, outcome, split=split * (1. - treatment_float), num_hidden_layers=2,
                extra_features=extra_features)

    tf.losses.add_loss(loss_ot0)
    tf.losses.add_loss(loss_ot1)
    tf.losses.add_loss(loss_t)

    training_loss = loss_ot0 + loss_ot1 + loss_t
    training_loss = training_loss

    outcome_st_treat = {'per_example_loss': per_example_loss_ot1,
                        'expectations': expectation_ot1}

    outcome_st_no_treat = {'per_example_loss': per_example_loss_ot0,
                           'expectations': expectation_ot0}

    treat = {'per_example_loss': per_example_loss_t,
             'logits': logits_t,
             'expectations': expectation_t}

    return training_loss, outcome_st_treat, outcome_st_no_treat, treat


def _create_or_get_dragonnet_bin_outcome(embedding, treatment, outcome, split, extra_features=None, getter=None):
    """
    Make predictions for the outcome, using the treatment and embedding,
    and predictions for the treatment, using the embedding
    Both outcome and treatment are assumed to be binary

    Note that we return the loss as a sum (and not a mean). This makes more sense for training dynamics

    Parameters
    ----------
    bert
    is_training
    treatment
    outcome
    label_dict
    split
    getter custom getter, for polyak averaging support

    Returns
    -------

    """

    treatment_float = tf.cast(treatment, tf.float32)
    with tf.variable_scope('dragon_net', reuse=tf.AUTO_REUSE, custom_getter=getter):
        with tf.variable_scope('treatment'):
            loss_t, per_example_loss_t, logits_t, expectation_t = _make_feedforward_classifier(
                embedding, treatment, 2, split, num_hidden_layers=0, extra_features=extra_features)

        with tf.variable_scope('outcome_st_treatment'):
            loss_ot1, per_example_loss_ot1, logits_ot1, expectation_ot1 = _make_feedforward_classifier(
                embedding, outcome, 2, split=split * treatment_float, num_hidden_layers=2,
                extra_features=extra_features)

        with tf.variable_scope('outcome_st_no_treatment'):
            loss_ot0, per_example_loss_ot0, logits_ot0, expectation_ot0 = _make_feedforward_classifier(
                embedding, outcome, 2, split=split * (1. - treatment_float), num_hidden_layers=2,
                extra_features=extra_features)

    tf.losses.add_loss(loss_ot0)
    tf.losses.add_loss(loss_ot1)
    tf.losses.add_loss(loss_t)

    training_loss = loss_ot0 + loss_ot1 + loss_t

    outcome_st_treat = {'per_example_loss': per_example_loss_ot1,
                        'logits': logits_ot1,
                        'expectations': expectation_ot1}

    outcome_st_no_treat = {'per_example_loss': per_example_loss_ot0,
                           'logits': logits_ot0,
                           'expectations': expectation_ot0}

    treat = {'per_example_loss': per_example_loss_t,
             'logits': logits_t,
             'expectations': expectation_t}

    return training_loss, outcome_st_treat, outcome_st_no_treat, treat


def _get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var  # if ema_var else var

    return ema_getter


def _binary_treatment_cont_outcome_eval_metric_fn(outcome, treatment, in_test,
                                                  per_example_loss_ot1,
                                                  per_example_loss_ot0,
                                                  per_example_loss_t, logits_t):
    treatment_float = tf.cast(treatment, tf.float32)

    ot1_dict = cont_label_eval_metric_fn(per_example_loss_ot1, outcome, split=treatment_float * in_test,
                                         family='outcome_given_treatment')
    ot0_dict = cont_label_eval_metric_fn(per_example_loss_ot0, outcome, split=(1 - treatment_float) * in_test,
                                         family='outcome_given_no_treatment')
    t_dict = binary_label_eval_metric_fn(per_example_loss_t, treatment, logits_t,
                                         split=in_test, family='treatment')

    return {**ot1_dict, **ot0_dict, **t_dict}


def binary_treat_cont_outcome_model_fn_builder(bert_config, init_checkpoint, learning_rate,
                                               num_train_steps, num_warmup_steps, use_tpu,
                                               use_one_hot_embeddings, label_pred=True, unsupervised=False,
                                               polyak=False, use_extra_features=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        treatment_name = params['treatment']
        outcome_name = params['outcome']

        treatment = features[treatment_name]
        outcome = features[outcome_name]

        # because reddit and peerread use slightly different text and pre-training structure
        if "op_token_ids" in features:
            token_ids = features["op_token_ids"]
            token_mask = features["op_token_mask"]
            maybe_masked_token_ids = features["op_maybe_masked_input_ids"]
        else:
            token_ids = features["token_ids"]
            token_mask = features["token_mask"]
            maybe_masked_token_ids = features["maybe_masked_input_ids"]

        if use_extra_features:
            extra_features_name = params['extra_features']
            extra_features = features[extra_features_name]
        else:
            extra_features = None

        #        subreddit_idx = features['subreddit']
        post_index = features['index']
        in_train = features['in_train']
        in_dev = features['in_dev']
        in_test = features['in_test']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss = 0

        if label_pred and not unsupervised:

            bert = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=token_ids,
                input_mask=token_mask,
                token_type_ids=None,
                use_one_hot_embeddings=use_one_hot_embeddings)

            bert_embedding = bert.get_pooled_output()

            label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_cont_outcome(bert_embedding, treatment, outcome, split=in_train,
                                                      extra_features=extra_features)
            total_loss = label_loss

        elif unsupervised and not label_pred:

            bert = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=maybe_masked_token_ids,
                input_mask=token_mask,
                token_type_ids=None,
                use_one_hot_embeddings=use_one_hot_embeddings)

            masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs = \
                _create_unsupervised_only_model(bert, bert_config, features)
            total_loss = masked_lm_loss

        elif unsupervised and label_pred:

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
            label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_cont_outcome(bert_embedding, treatment, outcome, split=in_train,
                                                      extra_features=extra_features)

            tf.losses.add_loss(masked_lm_loss)
            tf.losses.add_loss(0.1 * label_loss)

            tf.summary.scalar('masked_lm_loss', masked_lm_loss, family='loss')
            tf.summary.scalar('label_loss', label_loss, family='loss')

            total_loss = masked_lm_loss + 0.1 * label_loss

        else:
            raise ValueError('At least one of unsupervised or label_pred must be true')

        # some extras
        if label_pred:
            if polyak:
                print("Polyak")
                global_variables = tf.get_collection('trainable_variables',
                                                     'dragon_net')  # Get list of the global variables
                print('global_variables: {}'.format(global_variables))
                ema = tf.train.ExponentialMovingAverage(decay=0.998)
                ema_op = ema.apply(global_variables)

                polyak_getter = _get_getter(ema)

                label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                    _create_or_get_dragonnet_cont_outcome(bert_embedding, treatment, outcome, in_train,
                                                          getter=polyak_getter)

            # outcome given treatment
            treatment_float = tf.cast(treatment, tf.float32)

            make_label_regression_prediction_summaries(outcome_st_treat['per_example_loss'],
                                                       in_train * treatment_float,
                                                       "train-" + 'outcome_st_treat')

            make_label_regression_prediction_summaries(outcome_st_treat['per_example_loss'],
                                                       in_dev * treatment_float,
                                                       "dev-" + 'outcome_st_treat')

            # outcome given no treatment
            make_label_regression_prediction_summaries(outcome_st_no_treat['per_example_loss'],
                                                       in_train * (1 - treatment_float),
                                                       "train-" + 'outcome_st_no_treat')

            make_label_regression_prediction_summaries(outcome_st_no_treat['per_example_loss'],
                                                       in_dev * (1 - treatment_float),
                                                       "dev-" + 'outcome_st_no_treat')

            # treatment
            make_label_binary_prediction_summaries(treat['per_example_loss'],
                                                   treat['logits'],
                                                   treatment,
                                                   in_train,
                                                   "train-" + 'treat')

            make_label_binary_prediction_summaries(treat['per_example_loss'],
                                                   treat['logits'],
                                                   treatment,
                                                   in_dev,
                                                   "dev-" + 'treat')

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

            if polyak:
                with tf.control_dependencies([ema_op]):
                    train_op = optimization.create_optimizer(
                        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            else:
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            test_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_cont_outcome(bert_embedding, treatment, outcome, in_test,
                                                      extra_features=extra_features)

            eval_feed = {'outcome': outcome, 'treatment': treatment, 'in_test': in_test}
            eval_feed.update({'per_example_loss_ot1': outcome_st_treat['per_example_loss']})
            eval_feed.update({'per_example_loss_ot0': outcome_st_no_treat['per_example_loss']})
            eval_feed.update({'per_example_loss_t': treat['per_example_loss'],
                              'logits_t': treat['logits']})

            eval_metrics = (_binary_treatment_cont_outcome_eval_metric_fn, eval_feed)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=test_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:

            # remark: standard tensorflow workflow would be to only pass in testing data.
            # We're anticipating that all data may get passed in (due to possible relational structure)
            predictions = {'in_test': in_test}
            predictions.update({'treatment_probability': treat['expectations']})
            predictions.update(
                {'expected_outcome_st_treatment': outcome_st_treat['expectations']})
            predictions.update(
                {'expected_outcome_st_no_treatment': outcome_st_no_treat['expectations']})

            # need this info downstream... might as well save it here
            predictions.update({'outcome': outcome})
            predictions.update({'treatment': treatment})
            predictions.update({'index': post_index})
            #            predictions.update({'subreddit': subreddit_idx})

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def _binary_treatment_bin_outcome_eval_metric_fn(outcome, treatment, in_test,
                                                 per_example_loss_ot1, logits_ot1,
                                                 per_example_loss_ot0, logits_ot0,
                                                 per_example_loss_t, logits_t):
    treatment_float = tf.cast(treatment, tf.float32)

    ot1_dict = binary_label_eval_metric_fn(per_example_loss_ot1, outcome, logits_ot1,
                                           split=treatment_float * in_test, family='outcome_given_treatment')
    ot0_dict = binary_label_eval_metric_fn(per_example_loss_ot0, outcome, logits_ot0,
                                           split=(1 - treatment_float) * in_test, family='outcome_given_no_treatment')
    t_dict = binary_label_eval_metric_fn(per_example_loss_t, treatment, logits_t,
                                         split=in_test, family='treatment')

    return {**ot1_dict, **ot0_dict, **t_dict}


def binary_treat_binary_outcome_model_fn_builder(bert_config, init_checkpoint, learning_rate,
                                                 num_train_steps, num_warmup_steps, use_tpu,
                                                 use_one_hot_embeddings, label_pred=True, unsupervised=False,
                                                 polyak=False, use_extra_features=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        treatment_name = params['treatment']
        outcome_name = params['outcome']

        treatment = features[treatment_name]
        outcome = features[outcome_name]

        # because reddit and peerread use slightly different text and pre-training structure
        if "op_token_ids" in features:
            token_ids = features["op_token_ids"]
            token_mask = features["op_token_mask"]
            maybe_masked_token_ids = features["op_maybe_masked_input_ids"]
        else:
            token_ids = features["token_ids"]
            token_mask = features["token_mask"]
            maybe_masked_token_ids = features["maybe_masked_input_ids"]

        if use_extra_features:
            extra_features_name = params['extra_features']
            extra_features = features[extra_features_name]
        else:
            extra_features = None

        #        subreddit_idx = features['subreddit']
        post_index = features['index']
        in_train = features['in_train']
        in_dev = features['in_dev']
        in_test = features['in_test']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss = 0

        if label_pred and not unsupervised:

            bert = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=token_ids,
                input_mask=token_mask,
                token_type_ids=None,
                use_one_hot_embeddings=use_one_hot_embeddings)

            bert_embedding = bert.get_pooled_output()

            label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_bin_outcome(bert_embedding, treatment, outcome, split=in_train,
                                                     extra_features=extra_features)
            total_loss = label_loss

        elif unsupervised and not label_pred:

            bert = modeling.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=maybe_masked_token_ids,
                input_mask=token_mask,
                token_type_ids=None,
                use_one_hot_embeddings=use_one_hot_embeddings)

            masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs = \
                _create_unsupervised_only_model(bert, bert_config, features)
            total_loss = masked_lm_loss

        elif unsupervised and label_pred:

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
            label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_bin_outcome(bert_embedding, treatment, outcome, split=in_train,
                                                     extra_features=extra_features)

            tf.losses.add_loss(masked_lm_loss)
            tf.losses.add_loss(0.1 * label_loss)

            tf.summary.scalar('masked_lm_loss', masked_lm_loss, family='loss')
            tf.summary.scalar('label_loss', label_loss, family='loss')

            total_loss = masked_lm_loss + 0.1 * label_loss

        else:
            raise ValueError('At least one of unsupervised or label_pred must be true')

        # some extras
        if label_pred:
            if polyak:
                print("Polyak")
                global_variables = tf.get_collection('trainable_variables',
                                                     'dragon_net')  # Get list of the global variables
                print('global_variables: {}'.format(global_variables))
                ema = tf.train.ExponentialMovingAverage(decay=0.998)
                ema_op = ema.apply(global_variables)

                polyak_getter = _get_getter(ema)

                label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                    _create_or_get_dragonnet_cont_outcome(bert_embedding, treatment, outcome, in_train,
                                                          getter=polyak_getter)

            # outcome given treatment
            treatment_float = tf.cast(treatment, tf.float32)

            make_label_binary_prediction_summaries(outcome_st_treat['per_example_loss'],
                                                   outcome_st_treat['logits'],
                                                   outcome,
                                                   in_train * treatment_float,
                                                   "train-" + 'outcome_st_treat')

            make_label_binary_prediction_summaries(outcome_st_treat['per_example_loss'],
                                                   outcome_st_treat['logits'],
                                                   outcome,
                                                   in_dev * treatment_float,
                                                   "dev-" + 'outcome_st_treat')

            # outcome given no treatment
            make_label_binary_prediction_summaries(outcome_st_no_treat['per_example_loss'],
                                                   outcome_st_no_treat['logits'],
                                                   outcome,
                                                   in_train * (1 - treatment_float),
                                                   "train-" + 'outcome_st_no_treat')

            make_label_binary_prediction_summaries(outcome_st_no_treat['per_example_loss'],
                                                   outcome_st_no_treat['logits'],
                                                   outcome,
                                                   in_dev * (1 - treatment_float),
                                                   "dev-" + 'outcome_st_no_treat')

            # treatment
            make_label_binary_prediction_summaries(treat['per_example_loss'],
                                                   treat['logits'],
                                                   treatment,
                                                   in_train,
                                                   "train-" + 'treat')

            make_label_binary_prediction_summaries(treat['per_example_loss'],
                                                   treat['logits'],
                                                   treatment,
                                                   in_dev,
                                                   "dev-" + 'treat')

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

            if polyak:
                with tf.control_dependencies([ema_op]):
                    train_op = optimization.create_optimizer(
                        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            else:
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            test_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_cont_outcome(bert_embedding, treatment, outcome, in_test,
                                                      extra_features=extra_features)

            eval_feed = {'outcome': outcome, 'treatment': treatment, 'in_test': in_test}
            eval_feed.update({'per_example_loss_ot1': outcome_st_treat['per_example_loss']})
            eval_feed.update({'per_example_loss_ot0': outcome_st_no_treat['per_example_loss']})
            eval_feed.update({'per_example_loss_t': treat['per_example_loss'],
                              'logits_t': treat['logits']})

            eval_metrics = (_binary_treatment_bin_outcome_eval_metric_fn, eval_feed)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=test_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:

            # remark: standard tensorflow workflow would be to only pass in testing data.
            # We're anticipating that all data may get passed in (due to possible relational structure)
            predictions = {'in_test': in_test}
            predictions.update({'treatment_probability': treat['expectations']})
            predictions.update(
                {'expected_outcome_st_treatment': outcome_st_treat['expectations']})
            predictions.update(
                {'expected_outcome_st_no_treatment': outcome_st_no_treat['expectations']})

            # need this info downstream... might as well save it here
            predictions.update({'outcome': outcome})
            predictions.update({'treatment': treatment})
            predictions.update({'index': post_index})
            #            predictions.update({'subreddit': subreddit_idx})

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn
