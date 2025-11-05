# 📦 Import libraries
import pandas as pd
import tweepy
import re
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 🔐 Twitter API credentials
client = tweepy.Client(
    bearer_token='AAAAAAAAAAAAAAAAAAAAAFmX4wEAAAAA%2B6rDoxMk474MEV%2FZaiSmksVWavM%3DmVfVtMMZRBvav3o6Xdv5HhREgHxY89EBiPiBZI7yTJk4ZcADK1',
    consumer_key='aDfh5Lx9xCwXWZPOD2Pbrcffr',
    consumer_secret='SUqxW4ml0MR85mH9SybtVgu0p0KUnIhkWSTgEZYoG51JI8APJH',
    access_token='1979565204309704704-xZ5ZOz7MVmOinomTVjzl9Nf4QluUO3',
    access_token_secret='7FJzYBoR34BJzIIUKChKWVu35aspJPLfk4j5IbaYiBSHo',
    wait_on_rate_limit=False
)

# 🐦 Collect tweets
def get_tweets(keyword, max_items=50):
    print(f"\n🔍 Collecting tweets for: {keyword}")
    try:
        response = client.search_recent_tweets(
            query=f"{keyword} lang:en -is:retweet",
            tweet_fields=['id', 'created_at', 'public_metrics', 'text'],
            max_results=min(max_items, 100)
        )
        tweets = []
        if response.data:
            for tweet in response.data:
                tweets.append({
                    'tweet_id': tweet.id,
                    'time': tweet.created_at,
                    'tweet': tweet.text,
                    'retweet_count': tweet.public_metrics['retweet_count']
                })
        else:
            print(f"⚠️ No tweets found for '{keyword}'")
        return pd.DataFrame(tweets)
    except Exception as e:
        print(f"❌ Error fetching tweets for '{keyword}': {e}")
        return pd.DataFrame()

# 🧼 Clean text
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+|#|RT[\s]+|https?:\/\/\S+', '', str(text))
    return text.strip()

# 📈 Sentiment analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'

# 🔥 Buzz score
def compute_buzz(row):
    weight = {'positive': 1.5, 'neutral': 1.0, 'negative': 0.5}
    return row['retweet_count'] * weight.get(row['sentiment'], 1.0)

# 📊 Visualize sentiment
def plot_sentiment(df):
    counts = df['sentiment'].value_counts()
    if not counts.empty:
        counts.plot(kind='bar', color=['green', 'gray', 'red'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Tweet Count')
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No sentiment data to plot.")

# 🧠 ML prediction
def predict_sentiment(df):
    df['tweet_length'] = df['tweet'].apply(len)
    df['has_keywords'] = df['tweet'].apply(lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x))
    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])

    X = df[['tweet_length', 'has_keywords']]
    y = df['sentiment_label']

    if len(X) < 2:
        print("⚠️ Not enough data to train.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    X_test['predicted_sentiment'] = le.inverse_transform(y_pred)
    X_test['actual_sentiment'] = le.inverse_transform(y_test)
    X_test['tweet'] = df.loc[X_test.index, 'tweet']
    print("\n🔍 Sample Predictions:")
    print(X_test[['tweet', 'actual_sentiment', 'predicted_sentiment']])

# 🚀 Run pipeline
if __name__ == "__main__":
    keywords = ['Congress']
    all_data = pd.concat([get_tweets(kw, 50) for kw in keywords], ignore_index=True)

    if all_data.empty:
        print("⚠️ No data collected. Using mock data.")
        current_time = datetime.now()
        mock_data = [
            {'tweet_id': '1001', 'time': current_time, 'tweet': "PM Modi's rally drew crowds.", 'retweet_count': 1200},
            {'tweet_id': '1002', 'time': current_time, 'tweet': "Congress focused on youth jobs.", 'retweet_count': 850},
            {'tweet_id': '1003', 'time': current_time, 'tweet': "Election Commission updates.", 'retweet_count': 430},
        ]
        all_data = pd.DataFrame(mock_data)

    all_data['tweet'] = all_data['tweet'].apply(clean_text)
    all_data['sentiment'] = all_data['tweet'].apply(get_sentiment)
    all_data['buzz_score'] = all_data.apply(compute_buzz, axis=1)

    print("\n📊 Sentiment Counts:")
    print(all_data['sentiment'].value_counts())

    print("\n🔥 Top Buzz Tweets:")
    print(all_data.sort_values(by='buzz_score', ascending=False)[['tweet', 'buzz_score']].head(5))

    plot_sentiment(all_data)
    predict_sentiment(all_data)