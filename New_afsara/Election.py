import pandas as pd
import os
import re
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Configuration

manual_csv_path = 'political_tweets.csv'


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


# ML prediction
#
def run_ml_prediction(df):
    if df.empty or len(df) < 2:
        print("Not enough data for ML prediction.")
        return None, None

    # Lexicons
    positive_words = {'hope', 'growth', 'development', 'success', 'unity', 'inspiring', 'applaud', 'praised'}
    negative_words = {'crisis', 'corruption', 'scandal', 'protest', 'backlash', 'misuse', 'outrage', 'blames'}

    # Feature engineering
    df['tweet_length'] = df['tweet'].apply(len)
    df['has_keywords'] = df['tweet'].apply(lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x))
    df['num_words'] = df['tweet'].apply(lambda x: len(x.split()))
    df['contains_link'] = df['tweet'].apply(lambda x: int('http' in x))
    df['polarity'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['pos_word_count'] = df['tweet'].apply(lambda x: sum(word in positive_words for word in x.lower().split()))
    df['neg_word_count'] = df['tweet'].apply(lambda x: sum(word in negative_words for word in x.lower().split()))

    # Encode labels
    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])

    # Split
    X = df[['tweet', 'tweet_length', 'has_keywords', 'num_words', 'contains_link', 'polarity', 'pos_word_count', 'neg_word_count']]
    y = df['sentiment_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Column transformer
    preprocessor = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(max_features=100), 'tweet'),
        ('num', 'passthrough', ['tweet_length', 'has_keywords', 'num_words', 'contains_link', 'polarity', 'pos_word_count', 'neg_word_count'])
    ])

    # Pipeline
    pipeline = Pipeline([
        ('features', preprocessor),
        ('clf', RandomForestClassifier())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels = unique_labels(y_test, y_pred)
    target_names = le.inverse_transform(labels)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    X_test = X_test.copy()
    X_test['predicted_sentiment'] = le.inverse_transform(y_pred)
    X_test['actual_sentiment'] = le.inverse_transform(y_test)
    print("\nSample Predictions:")
    print(X_test[['tweet', 'actual_sentiment', 'predicted_sentiment']].head(10))

    return pipeline, le




def predict_sentiment_for_new_tweets(pipeline, le, new_tweets_df):
    positive_words = {'hope', 'growth', 'development', 'success', 'unity', 'inspiring', 'applaud', 'praised'}
    negative_words = {'crisis', 'corruption', 'scandal', 'protest', 'backlash', 'misuse', 'outrage', 'blames'}

    new_tweets_df['tweet'] = new_tweets_df['tweet'].apply(clean_text)
    new_tweets_df['tweet_length'] = new_tweets_df['tweet'].apply(len)
    new_tweets_df['has_keywords'] = new_tweets_df['tweet'].apply(lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x))
    new_tweets_df['num_words'] = new_tweets_df['tweet'].apply(lambda x: len(x.split()))
    new_tweets_df['contains_link'] = new_tweets_df['tweet'].apply(lambda x: int('http' in x))
    new_tweets_df['polarity'] = new_tweets_df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    new_tweets_df['pos_word_count'] = new_tweets_df['tweet'].apply(lambda x: sum(word in positive_words for word in x.lower().split()))
    new_tweets_df['neg_word_count'] = new_tweets_df['tweet'].apply(lambda x: sum(word in negative_words for word in x.lower().split()))

    X_new = new_tweets_df[['tweet', 'tweet_length', 'has_keywords', 'num_words', 'contains_link', 'polarity', 'pos_word_count', 'neg_word_count']]
    predicted_labels = pipeline.predict(X_new)
    predicted_sentiment = le.inverse_transform(predicted_labels)

    new_tweets_df['predicted_sentiment'] = predicted_sentiment
    print("\nPredicted Sentiment for New Tweets:")
    print(new_tweets_df[['tweet', 'predicted_sentiment']])

    new_tweets_df.to_csv('new_tweet_predictions.csv', index=False)
    print("\nPredictions saved to 'new_tweet_predictions.csv'")
    return new_tweets_df

# Main pipeline
def run_pipeline():
    cleaned_path = 'cleaned_df.csv'
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    else:
        df = load_manual_csv()

    df = df[['tweet_id', 'time', 'tweet', 'retweet_count']]
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype(int)
    df['tweet'] = df['tweet'].apply(clean_text)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)

    if os.path.exists(cleaned_path):
        existing_df = pd.read_csv(cleaned_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.drop_duplicates(subset='tweet_id', keep='last', inplace=True)
        combined_df.to_csv(cleaned_path, index=False)
        df = combined_df
    else:
        df.to_csv(cleaned_path, index=False)

    summarize_sentiment(df)
    model, le = run_ml_prediction(df)



            # Predict sentiment for new tweets
    new_tweets = pd.DataFrame({
        'tweet': [

        "PM Modi’s new initiative to support small farmers is receiving widespread appreciation across states.",
        "Congress spokesperson questioned the transparency of the newly launched economic policy.",
        "Citizens are hopeful that the government’s renewable energy plan will reduce pollution and create jobs",
        "Opposition parties slammed the administration for the delay in implementing welfare schemes..",
        "Congress launches campaign focused on rural development.",
        "BJP celebrates successful completion of highway project.",
        "Modi's silence on recent protests sparks debate online.",


            ]
    })
    predict_sentiment_for_new_tweets(model, le, new_tweets)


if __name__ == "__main__":
    run_pipeline()