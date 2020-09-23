# Introduction

This repository contains software and data for "Using Text Embeddings for Causal Inference" ([arxiv.org/abs/1905.12741](https://arxiv.org/abs/1905.12741)).
The paper describes a method for causal inference with text documents. For example, does adding a
theorem to a paper affect its chance of acceptance? The method adapts deep language models to address the causal problem. 

This software builds on
1. Bert: [github.com/google-research/bert](https://github.com/google-research/bert), and on
2. PeerRead: [github.com/allenai/PeerRead](https://github.com/allenai/PeerRead)

We include pre-processed PeerRead arxiv data for convenience.

There is also a [reference implementation in pytorch.](https://github.com/rpryzant/causal-bert-pytorch)

# Tensorflow 2
For new projects, we recommend building on the [reference tensorflow 2 implementation](https://github.com/vveitch/causal-text-embeddings-tf2).

# Requirements and setup

1. You'll need to download a pre-trained BERT model (following the above github link). We use `uncased_L-12_H-768_A-12`.
2. Install Tensorflow 1.12

# Data

1. We include a pre-processed copy of PeerRead data for convenience.
This data is a collection of arXiv papers submitted to computer science conferences, the accept/reject decisions for these papers,
and their abstracts.
The raw PeerRead data contains significantly more information.
You can get the raw data by following instructions at [github.com/allenai/PeerRead](https://github.com/allenai/PeerRead). 
Running the included pre-processing scripts in the PeerRead folder will recreate the included tfrecord file. 

2. The reddit data can be downloaded at [archive.org/details/reddit_posts_2018](https://archive.org/details/reddit_posts_2018).
This data includes all top-level reddit comments where the gender of the poster was annotated in some fashion.
Each post has meta information (score, date, username, etc.) and includes the text for the first reply.
The processed data used in the paper can be recreated by running the pre-processing scripts in the `reddit` folder.

You can also re-collect the data from Google BigQuery.
The SQL command to do this is in `reddit/data_cleaning/BigQuery_get_data`.
Modifying this script will allow you to change collection parameters (e.g., the year, which responses are included)


# Reproducing the PeerRead experiments

The default settings for the code match the settings used in the software.
These match the default settings used by BERT, except
1. we reduce batch size to allow training on a Titan X, and
2. we adjust the learning rate to account for this.

You'll run the from `src` code as 
`./PeerRead/submit_scripts/run_model.sh`
Before doing this, you'll need to edit `run_classifier.sh` to change 
`BERT_BASE_DIR=../../bert/pre-trained/uncased_L-12_H-768_A-12`
to
`BERT_BASE_DIR=[path to BERT_pre-trained]/uncased_L-12_H-768_A-12`.

The flag 
`--treatment=theorem_referenced`
controls the experiment. 
The flag 
`--simulated=real`
controls whether to use the real effect or one of the semi-synthetic modes.

The effect estimates can be reproduced by running `python -m result_processing.compute_ate`.
This takes in the predictions of the bert model (in tsv format) and passes them into downstream estimators
of the causal effect.

To reproduce the baselines, you'll need to produce a tsv for each simulated dataset you want to test on. To do this, you can run `python -m PeerRead.dataset.array_from_dataset` from src. The flag `--beta1=1.0` controls the strength of the confounding. (The other flags control other simulation parameters not used in the paper.)

# Misc.

The experiments in the paper use a version of BERT that was further pre-trained on the PeerRead corpus
using an unsupervised objective. 
This can be replicated with `./PeerRead/submit_scripts/run_classifier.sh`.
This takes about 24 hours on a single Titan Xp.
To use a pre-trained BERT, uncomment the `INIT_DIR` options in `run_classifier.sh`.

# Reproducing the Reddit experiment

1. First, get the data following instructions above and save it as `dat/reddit/2018.json`
2. Run data pre-processing with `python -m reddit.data_cleaning.process_reddit`
3. Once the data is processed, instructions for running the experiments are essentially the same as for PeerRead

# Maintainers
[Dhanya Sridhar](https://github.com/dsridhar91`) and [Victor Veitch](`github.com/vveitch`)

