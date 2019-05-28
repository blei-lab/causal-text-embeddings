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
import pandas as pd
import io
import json
from dateutil.parser import parse as parse_date
from PeerRead.ScienceParse.Paper import Paper

rng = random.Random(0)


def process_json_paper(paper_json_filename, scienceparse_dir, tokenizer):
    paper = Paper.from_json(paper_json_filename)
    return paper.ABSTRACT


def output_PeerRead_text(review_json_dir, parsedpdf_json_dir,
                        out_dir, out_file):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    paper_data = []
    print('Reading reviews from...', review_json_dir)
    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_json_dir)))
    for idx, paper_json_filename in enumerate(paper_json_filenames):
        paper = Paper.from_json(paper_json_filename)
        paper_data.append([paper.ID, paper.ABSTRACT])

    df = pd.DataFrame(paper_data, columns=['paper_id', 'abstract_text'])
    df.to_csv(out_dir + 'proc_abstracts.csv')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--review-json-dir', type=str, default='../dat/PeerRead/arxiv.all/all/reviews')
    parser.add_argument('--parsedpdf-json-dir', type=str, default='../dat/PeerRead/arxiv.all/all/parsed_pdfs')
    parser.add_argument('--out-dir', type=str, default='../dat/PeerRead/')
    parser.add_argument('--out-file', type=str, default='proc_text.csv')

    args = parser.parse_args()

    output_PeerRead_text(args.review_json_dir, args.parsedpdf_json_dir,
                           args.out_dir, args.out_file)


if __name__ == "__main__":
    main()
