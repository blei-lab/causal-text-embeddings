"""
create (hand-authored and lexical) features for baselines classifiers and save to under dataset folder in each split
"""

import sys, os, random, glob

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from PeerRead.ScienceParse.Paper import Paper
from PeerRead.ScienceParse.ScienceParseReader import ScienceParseReader


def get_PeerRead_hand_features(paper):
    sp = paper.get_scienceparse()

    hand_features = {}

    hand_features["accepted"] = paper.get_accepted()

    hand_features["most_recent_reference_year"] = sp.get_most_recent_reference_year() - 2000
    hand_features["num_recent_references"] = sp.get_num_recent_references(2017)
    hand_features["num_references"] = sp.get_num_references()
    hand_features["num_refmentions"] = sp.get_num_refmentions()
    hand_features["avg_length_reference_mention_contexts"] = sp.get_avg_length_reference_mention_contexts()

    hand_features["num_ref_to_figures"] = sp.get_num_ref_to_figures()
    hand_features["num_ref_to_tables"] = sp.get_num_ref_to_tables()
    hand_features["num_ref_to_sections"] = sp.get_num_ref_to_sections()

    hand_features["num_uniq_words"] = sp.get_num_uniq_words()
    hand_features["num_sections"] = sp.get_num_sections()
    hand_features["avg_sentence_length"] = sp.get_avg_sentence_length()

    hand_features["contains_appendix"] = sp.get_contains_appendix()

    hand_features["title_length"] = paper.get_title_len()
    hand_features["num_authors"] = sp.get_num_authors()
    hand_features["num_ref_to_equations"] = sp.get_num_ref_to_equations()
    hand_features["num_ref_to_theorems"] = sp.get_num_ref_to_theorems()

    abstract = str.lower(paper.ABSTRACT)
    hand_features["abstract_contains_deep"] = "deep" in abstract
    hand_features["abstract_contains_neural"] = "neural" in abstract
    hand_features["abstract_contains_embedding"] = "embedding" in abstract
    hand_features["abstract_contains_outperform"] = "outperform" in abstract
    hand_features["abstract_contains_novel"] = "novel" in abstract
    hand_features["abstract_contains_state-of-the-art"] = \
        "state-of-the-art" in abstract or "state of the art" in abstract

    title = str.lower(paper.TITLE)
    hand_features["title_contains_deep"] = "deep" in title
    hand_features["title_contains_neural"] = "neural" in title
    hand_features["title_contains_embedding"] = "embed" in title
    hand_features["title_contains_gan"] = ("gan" in title) or ("adversarial net" in title)

    return hand_features


def main(args):

    paper_json_dir = args[1]  # train/reviews
    scienceparse_dir = args[2]  # train/parsed_pdfs


    ################################
    # read reviews
    ################################
    print('Reading reviews from...', paper_json_dir)
    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(paper_json_dir)))
    papers = []
    for paper_json_filename in paper_json_filenames:
        paper = Paper.from_json(paper_json_filename)
        paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT,
                                                                   scienceparse_dir)
        papers.append(paper)
    random.shuffle(papers)
    print('Total number of reviews', len(papers))

    id = 1
    for p in papers:
        rec = int(p.get_accepted() == True)

        handy = get_PeerRead_hand_features(p)

        id += 1


if __name__ == "__main__":
    main(sys.argv)
