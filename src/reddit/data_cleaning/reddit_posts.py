import numpy as np
import pandas as pd
import os
import json
from scipy import sparse
from result_processing.helpers import tokenize_documents


# from plotnine import *

def load_term_counts(path='../dat/', force_redo=False):
    count_filename = path + 'reddit_term_counts'
    authors_counts_filename = path + 'reddit_author_term_counts'
    vocab_filename = path + 'vocab'

    if os.path.exists(count_filename + '.npz') and not force_redo:
        return sparse.load_npz(count_filename + '.npz'), sparse.load_npz(authors_counts_filename + '.npz'), np.load(
            vocab_filename + '.npy')

    reddit = load_reddit()
    post_docs = reddit['post_text'].values
    author_grouped = reddit.groupby('author')['post_text'].apply(lambda x: ' '.join(x)).reset_index()
    author_docs = author_grouped['post_text'].values
    counts, vocab, vec = tokenize_documents(post_docs)
    author_counts = vec.transform(author_docs)
    sparse.save_npz(count_filename, counts)
    sparse.save_npz(authors_counts_filename, author_counts)
    np.save(vocab_filename, vocab)
    return counts, author_counts, vocab


def convert_to_int_columns(df, exclude=['post_text', 'response_text', 'score', 'controversiality', 'gilded',
                                        'created_utc']):
    df = df.astype({'score': np.int64, 'controversiality': np.int64, 'gilded': np.int64, 'created_utc': np.int64})

    for col in df.columns.tolist():
        if col in exclude:
            continue
        df[col] = pd.Categorical(df[col]).codes

    return df


def subreddit_idx_to_subreddit(idx):
    """
    Warning: temporarily hardcoded for convenience. Beware!
    :param idx:
    :return:
    """

    subreddits = {0: '100DaysofKeto',
                  1: 'AskMen',
                  2: 'AskMenOver30',
                  3: 'AskWomen',
                  4: 'AskWomenOver30',
                  5: 'LGBTeens',
                  6: 'OkCupid',
                  7: 'Tinder',
                  8: 'childfree',
                  9: 'fatlogic',
                  10: 'financialindependence',
                  11: 'infertility',
                  12: 'infj',
                  13: 'keto',
                  14: 'loseit',
                  15: 'proED',
                  16: 'sexover30',
                  17: 'short',
                  18: 'tall',
                  19: 'xxketo'}

    return subreddits[idx]


def subreddit_male_prop(idx):
    subreddit = subreddit_idx_to_subreddit(idx)

    # lazy hardcoding
    gender_props = {'100DaysofKeto': 0.08290155440414508,
                    'AskMen': 0.9306885544915641,
                    'AskMenOver30': 0.9444306623666584,
                    'AskWomen': 0.053265121877821245,
                    'AskWomenOver30': 0.0836100211288862,
                    'LGBTeens': 0.9018952928382787,
                    'OkCupid': 0.6491243280735217,
                    'Tinder': 0.7985401459854015,
                    'childfree': 0.3436175847457627,
                    'fatlogic': 0.2293529255554572,
                    'financialindependence': 0.7604441360166551,
                    'infertility': 0.04929765886287625,
                    'infj': 0.6117755289788408,
                    'keto': 0.515695067264574,
                    'loseit': 0.24193122130091507,
                    'proED': 0.06660675582809114,
                    'sexover30': 0.5266344888108819,
                    'short': 0.875792872794372,
                    'tall': 0.8210111788617886,
                    'xxketo': 0.0022985674998973853}

    return gender_props[subreddit]


def subreddit_score_mean_and_std(idx):
    subreddit = subreddit_idx_to_subreddit(idx)

    means = {'100DaysofKeto': 4.16580310880829,
             'AskMen': 61.08446939321037,
             'AskMenOver30': 19.640205077317457,
             'AskWomen': 43.515366385795964,
             'AskWomenOver30': 16.83549652882584,
             'LGBTeens': 11.371757029672208,
             'OkCupid': 7.912259406970695,
             'Tinder': 54.55133819951338,
             'childfree': 41.629965572033896,
             'fatlogic': 61.86961140125697,
             'financialindependence': 21.741568355308814,
             'infertility': 4.269966555183946,
             'infj': 6.30328120208525,
             'keto': 6.335577664762253,
             'loseit': 10.759423210333322,
             'proED': 12.293228313157478,
             'sexover30': 7.862483545414656,
             'short': 6.966324530042671,
             'tall': 14.139481707317072,
             'xxketo': 6.622788654927554}

    stds = {'100DaysofKeto': 4.257771265407754,
            'AskMen': 358.85677169829677,
            'AskMenOver30': 62.022953833297635,
            'AskWomen': 203.28024082587,
            'AskWomenOver30': 26.279682061246756,
            'LGBTeens': 21.557311658010843,
            'OkCupid': 16.683316882877435,
            'Tinder': 329.3164620810436,
            'childfree': 105.1919761318333,
            'fatlogic': 115.06941068605869,
            'financialindependence': 70.252278798125,
            'infertility': 5.201437091946628,
            'infj': 9.37623701272285,
            'keto': 30.441002673493898,
            'loseit': 69.05652111583404,
            'proED': 20.152829147672076,
            'sexover30': 14.017546840438202,
            'short': 11.229113209821255,
            'tall': 33.3192418530642,
            'xxketo': 11.041996217862105}

    return means[subreddit], stds[subreddit]


def process_text_length(df):
    # mean words = 56.4 \pm 36.6
    df['post_length'] = df['post_text'].str.split(' ').str.len()
    return df


def load_reddit_processed(path='../dat/reddit/'):
    if os.path.exists(path + 'reddit_proc.csv'):
        return pd.read_csv(path + 'reddit_proc.csv', low_memory=False)

    reddit = load_reddit_latest(convert_columns=True)
    reddit.to_csv(path + 'reddit_proc.csv', index_label='orig_index')
    reddit['orig_index'] = reddit.index
    return reddit


def load_reddit_latest(path='../dat/', convert_columns=False):
    with open(os.path.join(path, '2018.json'), 'r') as f:
        record_dicts = []
        for line in f.readlines():
            record = json.loads(line)
            reply_list = record['reply']
            earliest_reply_text = None
            for reply_dict in sorted(reply_list, key=lambda x: x['created_utc']):
                if reply_dict['body'] != '[deleted]':
                    earliest_reply_text = reply_dict['body']
                if earliest_reply_text:
                    break

            if earliest_reply_text:
                record.pop('reply')
                record['response_text'] = earliest_reply_text
                record_dicts.append(record)

    reddit_df = pd.DataFrame(record_dicts)
    reddit_df = reddit_df[reddit_df.body != '[deleted]']
    reddit_df = reddit_df.rename(columns={'body': 'post_text'})

    if convert_columns:
        reddit_df = convert_to_int_columns(reddit_df)
    reddit_df = reddit_df.reset_index(drop=True)

    return reddit_df


def load_reddit(path='../dat/reddit/', convert_columns=False, use_latest=True):
    if use_latest:
        if path != '../dat/reddit/':
            return load_reddit_latest(path=path, convert_columns=convert_columns)
        else:
            return load_reddit_latest(convert_columns=convert_columns)

    posts_path = os.path.join(path, 'reddit_posts.csv')
    responses_path = os.path.join(path, 'reddit_responses.csv')
    posts = pd.read_csv(posts_path)
    responses = pd.read_csv(responses_path)
    reddit = posts.merge(responses, how='inner', on=['post_id', 'op_gender', 'op_id', 'subreddit'])

    unknown_indices = reddit['responder_gender_visible'] == 'unknown_gender'
    reddit['responder_gender_visible'][unknown_indices] = False

    if convert_columns:
        reddit = convert_to_int_columns(reddit)

    # TODO: text.count('<link>') 21524

    return reddit


def summarize_subreddit(subreddit, posts=False):
    male = subreddit.loc[subreddit['gender'] == 'male']
    female = subreddit.loc[subreddit['gender'] == 'female']

    if posts:
        print("POSTS \n")
        print("male:")
        print(male.score.describe())
        print("female:")
        print(female.score.describe())

        mscore = male.score.mean()
        fscore = female.score.mean()

    else:
        print("AUTHORS \n")
        print("male:")
        male_authors = male.groupby('author')['score'].agg(np.mean)
        print(male_authors.describe())
        mscore = male_authors.mean()

        # female
        print("female:")
        female_authors = female.groupby('author')['score'].agg(np.mean)
        print(female_authors.describe())
        fscore = female_authors.mean()

    return mscore, fscore


def reddit_summary(reddit):
    subreddit_names = reddit['subreddit'].unique()

    mscores = []
    fscores = []
    for subreddit_name in subreddit_names:
        subreddit = reddit.loc[reddit['subreddit'] == subreddit_name]
        print("*********************")
        print(subreddit_name)
        print("*********************")
        mscore, fscore = summarize_subreddit(subreddit)
        print("\n")

        mscores += [mscore]
        fscores += [fscore]

    # subreddit = reddit.loc[reddit['subreddit'] == 'financialindependence']
    print("*********************")
    print("Full")
    print("*********************")
    _, _ = summarize_subreddit(reddit, posts=True)
    print("\n")
    _, _ = summarize_subreddit(reddit, posts=False)

    print("\n")

    print("*********************")
    print("Average over subreddits")
    print("*********************")
    print("male: {}".format(np.mean(mscores)))
    print("female: {}".format(np.mean(fscores)))


def reddit_summary_simple(reddit):
    # print(reddit)

    reduced_reddit = reddit.groupby(['subreddit', 'gender', 'author']).agg(np.mean)
    topline = reduced_reddit.groupby(['subreddit', 'gender']).agg(np.mean)
    print("\n")
    print("Simple summary (aggregate over authors, then over subreddits and gender):")
    print(topline)

    naive_reduced = reddit.groupby(['subreddit', 'gender']).agg(np.mean)
    print("\n")
    print("Aggregate over subreddits and gender (not author):")
    print(naive_reduced)

    subreddit_reduced = reddit.groupby(['subreddit']).agg(np.mean)
    print("\n")
    print("Aggregate over subreddits:")
    print(subreddit_reduced)

    subreddit_counts = reddit.groupby(['subreddit']).size().reset_index(name='count')
    print("\n")
    print("Counts of posts per subreddit:")
    print(subreddit_counts)

    print("\n")
    print("Total posts:")
    print(reddit.shape[0])


def subreddit_author_summary(reddit, sub='keto'):
    if sub is not None:
        reddit = reddit[reddit.subreddit == sub]
    author_counts = reddit.groupby('author').size().reset_index(name='count')
    author_counts = author_counts.sort_values(by='count')
    print("\n")
    print("Author summary:")
    print(author_counts)


def main():
    reddit = load_reddit()
    reddit.score = reddit.score.astype(np.int)
    reddit.gender = reddit.gender.astype('category').cat.codes

    reduced = reddit.groupby('subreddit')['score'].agg(np.std)

    print(reddit.groupby('subreddit')['score'].agg(np.mean))
    print(reddit.groupby('subreddit')['score'].agg(np.std))

    # author_docs = reddit.groupby('author')['post_text'].apply(lambda x:' '.join(x)).reset_index()
    # print(author_docs)

    # reddit_summary(reddit)
    # reddit_summary_simple(reddit)
    # subreddit_author_summary(reddit,sub=None)


if __name__ == '__main__':
    main()
