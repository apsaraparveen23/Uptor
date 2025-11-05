import pandas as pd
import tweepy
import os
import re
import time
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Configuration
manual_csv_path = 'manual_tweets.csv'
keywords = ['BJP', 'Congress']


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
    csv_files = [f for f in os.listdir('../Afsara_project') if f.endswith('.csv')]
    if not csv_files:
        return load_manual_csv()

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


# Load manual CSV fallback
def load_manual_csv():
    if not os.path.exists(manual_csv_path):
        print("Manual CSV not found. Loading mock data...")
        return load_mock_data()
    df = pd.read_csv(manual_csv_path)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype(int)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)
    df.to_csv('cleaned_df.csv', index=False)
    return df


# Load mock data
def load_mock_data():
    current_time = datetime.now()
    mock_data = [
        {'tweet_id': '1001', 'time': current_time, 'tweet': "PM Modi's rally in Meerut drew massive crowds.",
         'retweet_count': 1200},
        {'tweet_id': '1002', 'time': current_time,
         'tweet': "Rahul Gandhi focused on youth employment in his Ramlila speech.", 'retweet_count': 850},
        {'tweet_id': '1003', 'time': current_time, 'tweet': "Election Commission announces new voting guidelines.",
         'retweet_count': 430},
        {'tweet_id': '1004', 'time': current_time, 'tweet': "BJP campaigns in Varanasi with promises of development.",
         'retweet_count': 670},
        {'tweet_id': '1005', 'time': current_time, 'tweet': "Congress accuses BJP of misusing government machinery.",
         'retweet_count': 520}
    ]
    df = pd.DataFrame(mock_data)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)
    df.to_csv('cleaned_df.csv', index=False)
    return df


# Sentiment summary and visualization
def summarize_sentiment(df):
    print("\nSentiment Distribution:")
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

    print(
        f"\nElection Buzz Summary:\n- Positive tweets: {sentiment_counts.get('positive', 0)}\n- Neutral tweets: {sentiment_counts.get('neutral', 0)}\n- Negative tweets: {sentiment_counts.get('negative', 0)}")


# Feature engineering + ML prediction
def run_ml_prediction(df):
    if df.empty or len(df) < 2:
        print("Not enough data for ML prediction.")
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
    # print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    X_test = X_test.copy()
    X_test['predicted_sentiment'] = le.inverse_transform(y_pred)
    X_test['actual_sentiment'] = le.inverse_transform(y_test)
    X_test['tweet'] = df.loc[X_test.index, 'tweet']
    print("\nSample Predictions:")
    print(X_test[['tweet', 'actual_sentiment', 'predicted_sentiment']].head(10))


# Main pipeline

def run_pipeline():
    # Load existing cleaned data or fallback to manual/mock
    cleaned_path = 'cleaned_df.csv'
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    else:
        df = load_manual_csv()

    # Clean and enrich the data
    df = df[['tweet_id', 'time', 'tweet', 'retweet_count']]
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype(int)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)

    # ✅ Append new data to cleaned_df.csv
    if os.path.exists(cleaned_path):
        existing_df = pd.read_csv(cleaned_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.drop_duplicates(subset='tweet_id', keep='last', inplace=True)
        combined_df.to_csv(cleaned_path, index=False)
        df = combined_df
    else:
        df.to_csv(cleaned_path, index=False)

    summarize_sentiment(df)
    run_ml_prediction(df)

if __name__ == "__main__":
    run_pipeline()

