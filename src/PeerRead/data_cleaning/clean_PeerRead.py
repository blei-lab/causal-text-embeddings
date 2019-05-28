import argparse
import os

import bert.tokenization as tokenization
from PeerRead.data_cleaning.process_PeerRead_abstracts import clean_PeerRead_dataset

dataset_names = ['acl_2017',
                  'arxiv.cs.ai_2007-2017',
                  'arxiv.cs.cl_2007-2017',
                  'arxiv.cs.lg_2007-2017',
                  'conll_2016',
                  'iclr_2017',
                  'nips_2013',
                  'nips_2014',
                  'nips_2015',
                  'nips_2016',
                  'nips_2017'
                  ]

dataset_paths = ['acl_2017',
                  'arxiv.cs.ai_2007-2017',
                  'arxiv.cs.cl_2007-2017',
                  'arxiv.cs.lg_2007-2017',
                  'conll_2016',
                  'iclr_2017',
                  'nips_2013-2017/2013',
                  'nips_2013-2017/2014',
                  'nips_2013-2017/2015',
                  'nips_2013-2017/2016',
                  'nips_2013-2017/2017'
                  ]

dataset_paths = dict(zip(dataset_names, dataset_paths))

dataset_years = {'acl_2017': 2017,
                  'conll_2016': 2016,
                  'iclr_2017': 2017,
                  'arxiv.cs.ai_2007-2017': None,
                  'arxiv.cs.cl_2007-2017': None,
                  'arxiv.cs.lg_2007-2017': None,
                  'nips_2013': 2013,
                  'nips_2014': 2014,
                  'nips_2015': 2015,
                  'nips_2016': 2016,
                  'nips_2017': 2017}

# dataset_venues = {k: v for v,k in enumerate(dataset_names)}

dataset_venues = {'acl_2017': 0,
                  'conll_2016': 1,
                  'iclr_2017': 2,
                  'nips_2013': 3,
                  'nips_2014': 3,
                  'nips_2015': 3,
                  'nips_2016': 3,
                  'nips_2017': 3,
                  'arxiv.cs.ai_2007-2017': 4,
                  'arxiv.cs.cl_2007-2017': 5,
                  'arxiv.cs.lg_2007-2017': 6,
                  }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-dir', type=str, default='../dat/PeerRead')
    parser.add_argument('--vocab-file', type=str, default='../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt')
    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=True)

    def proc_dataset(dataset):
        all_dir = os.path.join(datasets_dir, dataset_paths[dataset], 'all')
        review_json_dir = os.path.join(all_dir, 'reviews')
        parsedpdf_json_dir = os.path.join(all_dir, 'parsed_pdfs')

        venue = dataset_venues[dataset]
        year = dataset_years[dataset]

        out_dir = os.path.join(datasets_dir, 'proc')
        out_file = dataset + '.tf_record'
        max_abs_len = 250

        clean_PeerRead_dataset(review_json_dir, parsedpdf_json_dir, venue, year, out_dir, out_file, max_abs_len,
                               tokenizer)

    # pool = mp.Pool(4)
    # pool.map(proc_dataset, dataset_names)

    for dataset in dataset_names:
        proc_dataset(dataset)


if __name__ == "__main__":
    main()
