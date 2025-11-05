# 📦 Import required libraries
import pandas as pd
import tweepy
import os
import re
import time
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
use_manual_csv = True   # ✅ Change to False if you want to collect live tweets
manual_csv_path = 'manual_tweets.csv'
# 🔐 Twitter API v2 credentials
bearer_token = 'YOUR_BEARER_TOKEN'
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# 🧵 Initialize Tweepy Client
client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

# 🐦 Function to collect tweets and save CSV
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
        print(f"⚠ Rate limit exceeded for keyword '{keyword}'. Saving empty CSV.")

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

# 🔍 Collect tweets
keywords = ['BJP', 'Congress']
for keyword in keywords:
    get_related_tweets(keyword, max_items=10)
    time.sleep(5)

# 🧼 Text cleaning function
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[:;\n_1]', '', text)
    return text

# 😊 Sentiment analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# 🔥 Buzz score
def compute_buzz(row):
    weight = {'positive': 1.5, 'neutral': 1.0, 'negative': 0.5}
    return row['retweet_count'] * weight.get(row['sentiment'], 1.0)

# 📊 Combine CSVs
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if csv_files:
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined_df = combined_df[['tweet_id', 'time', 'tweet', 'retweet_count']]
    combined_df.dropna(subset=['tweet_id'], inplace=True)
    combined_df['time'] = combined_df['time'].ffill().bfill()
    combined_df['retweet_count'] = combined_df['retweet_count'].fillna(0).astype(int)
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df.drop_duplicates(subset='tweet_id', keep='last', inplace=True)

    combined_df['tweet'] = combined_df['tweet'].apply(cleanTxt)
    combined_df['sentiment'] = combined_df['tweet'].apply(get_sentiment)
    combined_df['buzz_score'] = combined_df.apply(compute_buzz, axis=1)
    combined_df.to_csv('cleaned_df.csv', index=False)
    print("\n✅ Cleaned dataset with sentiment and buzz score saved as 'cleaned_df.csv'")

else:
    print("⚠️ No CSV files found. Loading mock data...")
    current_time = datetime.now()
    mock_data = [
        {'tweet_id': '1001', 'time': current_time, 'tweet': "PM Modi's rally in Meerut drew massive crowds.", 'retweet_count': 1200},
        {'tweet_id': '1002', 'time': current_time, 'tweet': "Rahul Gandhi focused on youth employment in his Ramlila speech.", 'retweet_count': 850},
        {'tweet_id': '1003', 'time': current_time, 'tweet': "Election Commission announces new voting guidelines.", 'retweet_count': 430},
        {'tweet_id': '1004', 'time': current_time, 'tweet': "BJP campaigns in Varanasi with promises of development.", 'retweet_count': 670},
        {'tweet_id': '1005', 'time': current_time, 'tweet': "Congress accuses BJP of misusing government machinery.", 'retweet_count': 520}
    ]
    combined_df = pd.DataFrame(mock_data)
    combined_df['tweet'] = combined_df['tweet'].apply(cleanTxt)
    combined_df['sentiment'] = combined_df['tweet'].apply(get_sentiment)
    combined_df['buzz_score'] = combined_df.apply(compute_buzz, axis=1)
    combined_df.to_csv('cleaned_df.csv', index=False)
    print("✅ Mock dataset with sentiment and buzz score saved as 'cleaned_df.csv'")

# 📈 Sentiment Distribution
print("\n Sentiment Distribution:")
print(combined_df['sentiment'].value_counts())

# 🧾 Sample tweets by sentiment
print("\n Positive Tweets:")
print(combined_df[combined_df['sentiment'] == 'positive'][['time', 'tweet']].head(3))

print("\n Neutral Tweets:")
print(combined_df[combined_df['sentiment'] == 'neutral'][['time', 'tweet']].head(3))

print("\n Negative Tweets:")
print(combined_df[combined_df['sentiment'] == 'negative'][['time', 'tweet']].head(3))

# 🔥 Top buzzworthy tweets
print("\n Top Buzz Tweets:")
print(combined_df.sort_values(by='buzz_score', ascending=False)[['time', 'tweet', 'buzz_score']].head(5))

# 📊 Bar chart of sentiment
sentiment_counts = combined_df['sentiment'].value_counts()
if not sentiment_counts.empty:
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No sentiment data available to plot.")

# 🗳️ Carousel-style summary
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

# 🔍 Feature Engineering + DBSCAN Clustering
if combined_df.empty:
    print("⚠️ No data available for clustering. Skipping DBSCAN.")
else:
    combined_df['tweet_length'] = combined_df['tweet'].apply(len)
    combined_df['has_keywords'] = combined_df['tweet'].apply(lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x))

    # Prepare features
    X = combined_df[['tweet_length', 'has_keywords']].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN
    db = DBSCAN(eps=0.5, min_samples=3)
    combined_df['cluster'] = db.fit_predict(X_scaled)

    # Show cluster summary
    print("\n🔍 DBSCAN Cluster Summary:")
    cluster_counts = combined_df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"Noise points: {count}")
        else:
            print(f"Cluster {cluster_id}: {count} tweets")

    # Show sample tweets from each cluster
    for cluster_id in sorted(combined_df['cluster'].unique()):
        print(f"\n🧵 Sample tweets from Cluster {cluster_id}:")
        print(combined_df[combined_df['cluster'] == cluster_id][['tweet', 'sentiment']].head(3))

    # Optional: Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=combined_df['cluster'], cmap='rainbow', s=60, edgecolor='k')
plt.xlabel('Tweet Length (scaled)')
plt.ylabel('Has Keywords (scaled)')
plt.title('DBSCAN Clustering of Tweets')
plt.grid(True)
plt.show()
