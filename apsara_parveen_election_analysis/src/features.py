from textblob import TextBlob


def add_features(df):
    positive_words = {'hope', 'growth', 'development', 'success', 'unity', 'inspiring', 'applaud', 'praised'}
    negative_words = {'crisis', 'corruption', 'scandal', 'protest', 'backlash', 'misuse', 'outrage', 'blames'}

    df['tweet_length'] = df['tweet'].apply(len)
    df['has_keywords'] = df['tweet'].apply(lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x))
    df['num_words'] = df['tweet'].apply(lambda x: len(x.split()))
    df['contains_link'] = df['tweet'].apply(lambda x: int('http' in x))
    df['polarity'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['pos_word_count'] = df['tweet'].apply(lambda x: sum(word in positive_words for word in x.lower().split()))
    df['neg_word_count'] = df['tweet'].apply(lambda x: sum(word in negative_words for word in x.lower().split()))
    return df
