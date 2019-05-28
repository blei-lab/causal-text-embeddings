"""
vv: wrote this to inspect what bert's tokenizer does with vocabulary terms it doesn't know.
The answer is: it splits them into word pieces where it has embeddings for each piece. Example:

tokenizer.tokenize('embedding')
['em', '##bed', '##ding']

tokenizer.convert_tokens_to_ids(['em', '##bed', '##ding'])
[7861, 8270, 4667]

Accordingly, the meaning of embedding can be learned so long as there's a suitably rich training corpus
"""

import argparse
import glob
import random

import io
import json

import bert.tokenization as tokenization

rng = random.Random(0)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--review-json-dir', type=str, default=None)
    parser.add_argument('--vocab-file', type=str, default=None)

    args = parser.parse_args()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=True)

    review_json_dir = args.review_json_dir

    print('Reading reviews from...', review_json_dir)
    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_json_dir)))

    paper_json_filename = paper_json_filenames[0]
    with io.open(paper_json_filename) as json_file:
        loaded = json.load(json_file)
    abstract = loaded['abstract']
    print(abstract)
    tokens = tokenizer.tokenize(abstract)
    print(tokens)
    print(tokenizer.convert_tokens_to_ids(tokens))

    # for idx, paper_json_filename in enumerate(paper_json_filenames):
    #     with io.open(paper_json_filename) as json_file:
    #         loaded = json.load(json_file)
    #
    #     print(loaded['abstract'])


if __name__ == "__main__":
    main()
