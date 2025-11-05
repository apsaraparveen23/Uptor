# 📦 Import required libraries
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

# -------------------------------------------
#CONFIGURATION
# -------------------------------------------
use_manual_csv = True   # ✅ Change to False if you want to collect live tweets
manual_csv_path = 'manual_tweets.csv'  # Your 10,000-record CSV file path

# -------------------------------------------
# 🔐 Twitter API v2 credentials
# (only needed if use_manual_csv = False)
# -------------------------------------------
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAfy4wEAAAAAfT%2FtzO0zo5a51egjGBlvo%2BQ%2B22k%3DvzhM9EIHmKCyiGeXtCNfXiWKVmml2WoNa9QYpQ9xDY28SH2pud'
api_key = 'PBmJ6cIVEU8ctG1CM8ZCIIZLV'
api_secret = 'SUqxW4ml0MR85mH9SybtVgu0p0KUnIhkWSTgEZYoG51JI8APJH'
access_token = '1981392334001754112-f1d947RkSKJtcB238gjM29oN1MeeN5'
access_token_secret = 'CZUQnvqA9jQwitnlQMkUAvtpa5XypLckhaJltKA2NrQjT'

#  Initialize Tweepy Client (only used if collecting live tweets)
if not use_manual_csv:
    client = tweepy.Client(
        bearer_token=bearer_token,
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        wait_on_rate_limit=True
    )

# -------------------------------------------
#  Function to collect tweets from Twitter
# -------------------------------------------
def get_related_tweets(keyword, max_items=50):
    tweet_id = []
    tweet_time = []
    tweet_text = []
    retweet_count = []

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
            print(f"⚠️ No tweets found for keyword '{keyword}'.")

    except tweepy.TooManyRequests:
        print(f"⚠️ Rate limit exceeded for keyword '{keyword}'. Saving empty CSV.")

    except Exception as e:
        print(f"❌ Error for keyword '{keyword}': {e}")

    df = pd.DataFrame({
        'tweet_id': tweet_id,
        'time': tweet_time,
        'tweet': tweet_text,
        'retweet_count': retweet_count
    })

    filename = f"{keyword.replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"📁 Saved {len(df)} tweets for '{keyword}' to {filename}")
    return df

# -------------------------------------------
# 🧾 LOAD MANUAL CSV INSTEAD OF TWITTER DATA
# -------------------------------------------
if use_manual_csv and os.path.exists(manual_csv_path):
    print(f"📥 Loading manually provided tweets from {manual_csv_path} ...")
    combined_df = pd.read_csv(manual_csv_path)

    required_cols = {'tweet_id', 'time', 'tweet', 'retweet_count'}
    if not required_cols.issubset(combined_df.columns):
        raise ValueError(f"❌ CSV missing required columns: {required_cols}")

    combined_df['time'] = pd.to_datetime(combined_df['time'], errors='coerce')
    combined_df['retweet_count'] = combined_df['retweet_count'].fillna(0).astype(int)

else:
    print("⚠️ Manual CSV not found or disabled — collecting tweets from Twitter API.")
    keywords = ['BJP', 'Congress']
    for keyword in keywords:
        get_related_tweets(keyword, max_items=10)
        time.sleep(5)

    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# -------------------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------------------
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[:;\n_1]', '', text)
    return text

# -------------------------------------------
#  SENTIMENT ANALYSIS FUNCTION
# -------------------------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# -------------------------------------------
#  BUZZ SCORE FUNCTION
# -------------------------------------------
def compute_buzz(row):
    weight = {'positive': 1.5, 'neutral': 1.0, 'negative': 0.5}
    return row['retweet_count'] * weight.get(row['sentiment'], 1.0)

# -------------------------------------------
#  CLEAN AND ENRICH DATA
# -------------------------------------------
combined_df['tweet'] = combined_df['tweet'].apply(cleanTxt)
combined_df['sentiment'] = combined_df['tweet'].apply(get_sentiment)
combined_df['buzz_score'] = combined_df.apply(compute_buzz, axis=1)
combined_df.to_csv('cleaned_df.csv', index=False)
print("\n Cleaned dataset with sentiment and buzz score saved as 'cleaned_df.csv'")

# -------------------------------------------
#  SENTIMENT DISTRIBUTION
# -------------------------------------------
print("\n Sentiment Distribution:")
print(combined_df['sentiment'].value_counts())

print("\n🔥 Top Buzz Tweets:")
print(combined_df.sort_values(by='buzz_score', ascending=False)[['time', 'tweet', 'buzz_score']].head(5))

#  Bar chart of sentiment
sentiment_counts = combined_df['sentiment'].value_counts()
if not sentiment_counts.empty:
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.tight_layout()
    plt.show()

# -------------------------------------------
# 🎨 SUMMARY
# -------------------------------------------
positive = sentiment_counts.get('positive', 0)
neutral = sentiment_counts.get('neutral', 0)
negative = sentiment_counts.get('negative', 0)

summary = f"""
🗳️ Election Buzz Summary:
- 😊 Positive tweets: {positive}
- 😐 Neutral tweets: {neutral}
- 😠 Negative tweets: {negative}

Top buzz topics include rallies, speeches, and Election Commission updates.
"""
print(summary)

# -------------------------------------------
#  FEATURE ENGINEERING + ML PREDICTION
# -------------------------------------------
if combined_df.empty:
    print("⚠️ No data available for training. Skipping ML prediction.")
else:
    combined_df['tweet_length'] = combined_df['tweet'].apply(len)
    combined_df['has_keywords'] = combined_df['tweet'].apply(
        lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x)
    )



    le = LabelEncoder()
    combined_df['sentiment_label'] = le.fit_transform(combined_df['sentiment'])

    X = combined_df[['tweet_length', 'has_keywords']]
    y = combined_df['sentiment_label']

    if len(X) < 2:
        print("⚠️ Not enough data for train-test split.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        X_test = X_test.copy()
        X_test['predicted_sentiment'] = le.inverse_transform(y_pred)
        X_test['actual_sentiment'] = le.inverse_transform(y_test)
        X_test['tweet'] = combined_df.loc[X_test.index, 'tweet']
        print("\n🔍 Sample Predictions:")
        print(X_test[['tweet', 'actual_sentiment', 'predicted_sentiment']].head(10))
