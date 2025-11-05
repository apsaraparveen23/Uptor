import re
import pandas as pd
from datetime import datetime


def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[:;\n_1]', '', text)
    return text


def load_mock_data():
    from .sentiment import get_sentiment, compute_buzz
    current_time = datetime.now()
    mock_data = [
        {'tweet_id': '1001', 'time': current_time, 'tweet': "PM Modi's rally in Meerut drew massive crowds.", 'retweet_count': 1200},
        {'tweet_id': '1002', 'time': current_time, 'tweet': "Rahul Gandhi focused on youth employment in his Ramlila speech.", 'retweet_count': 850},
        {'tweet_id': '1003', 'time': current_time, 'tweet': "Election Commission announces new voting guidelines.", 'retweet_count': 430},
        {'tweet_id': '1004', 'time': current_time, 'tweet': "BJP campaigns in Varanasi with promises of development.", 'retweet_count': 670},
        {'tweet_id': '1005', 'time': current_time, 'tweet': "Congress accuses BJP of misusing government machinery.", 'retweet_count': 520},
    ]
    df = pd.DataFrame(mock_data)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)
    df.to_csv('data/cleaned_df.csv', index=False)
    return df


def load_manual_csv(path='data/political_tweets.csv'):
    from .sentiment import get_sentiment, compute_buzz
    import os

    if not os.path.exists(path):
        print("Manual CSV not found. Loading mock data...")
        return load_mock_data()

    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype(int)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)
    df.to_csv('data/cleaned_df.csv', index=False)
    return df
