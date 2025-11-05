import pandas as pd
import tweepy
import os
import re
import time
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

use_manual_csv = True   # ✅ Change to False if you want to collect live tweets
manual_csv_path = 'manual_tweets.csv'  # Your 10,000-record CSV file path

# Twitter API credentials
def initialize_twitter_client():
    return tweepy.Client(
        bearer_token='AAAAAAAAAAAAAAAAAAAAAAfy4wEAAAAAfT%2FtzO0zo5a51egjGBlvo%2BQ%2B22k%3DvzhM9EIHmKCyiGeXtCNfXiWKVmml2WoNa9QYpQ9xDY28SH2pud',
        consumer_key='PBmJ6cIVEU8ctG1CM8ZCIIZLV',
        consumer_secret='SUqxW4ml0MR85mH9SybtVgu0p0KUnIhkWSTgEZYoG51JI8APJH',
        access_token='1981392334001754112-f1d947RkSKJtcB238gjM29oN1MeeN5',
        access_token_secret='CZUQnvqA9jQwitnlQMkUAvtpa5XypLckhaJltKA2NrQjT',
        wait_on_rate_limit=True
    )

# Collect tweets for a keyword
def get_related_tweets(client, keyword, max_items=50):
    tweet_id, tweet_time, tweet_text, retweet_count = [], [], [], []

    try:
        response = client.search_recent_tweets(
            query=keyword,
            tweet_fields=['id', 'created_at', 'public_metrics', 'text'],
            max_results=min(max_items, 100)
        )
        if response.data:
            for tweet in response.data:
                tweet_id.append(tweet.id)
                tweet_time.append(tweet.created_at)
                tweet_text.append(tweet.text)
                retweet_count.append(tweet.public_metrics['retweet_count'])
        else:
            print(f"No tweets found for keyword '{keyword}'")
    except tweepy.TooManyRequests:
        print(f"Rate limit exceeded for keyword '{keyword}'")
    except Exception as e:
        print(f"Error for keyword '{keyword}': {e}")

    df = pd.DataFrame({
        'tweet_id': tweet_id,
        'time': tweet_time,
        'tweet': tweet_text,
        'retweet_count': retweet_count
    })
    df.to_csv(f"{keyword.replace(' ', '_')}.csv", index=False)
    return df

# Clean tweet text
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[:;\n_1]', '', text)
    return text

# Sentiment analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# Buzz score calculation
def compute_buzz(row):
    weight = {'positive': 1.5, 'neutral': 1.0, 'negative': 0.5}
    return row['retweet_count'] * weight.get(row['sentiment'], 1.0)

# Load and clean CSVs
def load_and_clean_data():
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        return load_mock_data()

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df = df[['tweet_id', 'time', 'tweet', 'retweet_count']]
    df.dropna(subset=['tweet_id'], inplace=True)
    df['time'] = pd.to_datetime(df['time'].ffill().bfill())
    df['retweet_count'] = df['retweet_count'].fillna(0).astype(int)
    df.drop_duplicates(subset='tweet_id', keep='last', inplace=True)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)
    df.to_csv('cleaned_df.csv', index=False)
    return df

# Load mock data
def load_mock_data():
    current_time = datetime.now()
    mock_data = [
        {'tweet_id': '1001', 'time': current_time, 'tweet': "PM Modi's rally in Meerut drew massive crowds.", 'retweet_count': 1200},
        {'tweet_id': '1002', 'time': current_time, 'tweet': "Rahul Gandhi focused on youth employment in his Ramlila speech.", 'retweet_count': 850},
        {'tweet_id': '1003', 'time': current_time, 'tweet': "Election Commission announces new voting guidelines.", 'retweet_count': 430},
        {'tweet_id': '1004', 'time': current_time, 'tweet': "BJP campaigns in Varanasi with promises of development.", 'retweet_count': 670},
        {'tweet_id': '1005', 'time': current_time, 'tweet': "Congress accuses BJP of misusing government machinery.", 'retweet_count': 520}
    ]
    df = pd.DataFrame(mock_data)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)
    df.to_csv('cleaned_df.csv', index=False)
    return df

# Display sentiment distribution and top tweets
def summarize_sentiment(df):
    print("Sentiment Distribution:")
    print(df['sentiment'].value_counts())

    print("\nPositive Tweets:")
    print(df[df['sentiment'] == 'positive'][['time', 'tweet']].head(3))

    print("\nNeutral Tweets:")
    print(df[df['sentiment'] == 'neutral'][['time', 'tweet']].head(3))

    print("\nNegative Tweets:")
    print(df[df['sentiment'] == 'negative'][['time', 'tweet']].head(3))

    print("\nTop Buzz Tweets:")
    print(df.sort_values(by='buzz_score', ascending=False)[['time', 'tweet', 'buzz_score']].head(5))

    sentiment_counts = df['sentiment'].value_counts()
    if not sentiment_counts.empty:
        sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.tight_layout()
        plt.show()

    print(f"\nElection Buzz Summary:\n- Positive tweets: {sentiment_counts.get('positive', 0)}\n- Neutral tweets: {sentiment_counts.get('neutral', 0)}\n- Negative tweets: {sentiment_counts.get('negative', 0)}")

# Train and evaluate ML model
def train_sentiment_model(df):
    if df.empty or len(df) < 2:
        print("Not enough data for training.")
        return

    df['tweet_length'] = df['tweet'].apply(len)
    df['has_keywords'] = df['tweet'].apply(lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x))

    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])

    X = df[['tweet_length', 'has_keywords']]
    y = df['sentiment_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    X_test = X_test.copy()
    X_test['predicted_sentiment'] = le.inverse_transform(y_pred)
    X_test['actual_sentiment'] = le.inverse_transform(y_test)
    X_test['tweet'] = df.loc[X_test.index, 'tweet']
    print("\nSample Predictions:")
    print(X_test[['tweet', 'actual_sentiment', 'predicted_sentiment']].head(10))

# Main procedure
def run_pipeline():
    client = initialize_twitter_client()
    keywords = ['BJP', 'Congress']
    for keyword in keywords:
        get_related_tweets(client, keyword, max_items=10)
        time.sleep(5)

    df = load_and_clean_data()
    summarize_sentiment(df)
    train_sentiment_model(df)

# Run the pipeline
run_pipeline()