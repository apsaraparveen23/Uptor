# 📦 Import required libraries
import pandas as pd
import tweepy
import os
import re
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt


# 🔐 Twitter API v2 credentials (replace with your actual values)
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAfy4wEAAAAAfT%2FtzO0zo5a51egjGBlvo%2BQ%2B22k%3DvzhM9EIHmKCyiGeXtCNfXiWKVmml2WoNa9QYpQ9xDY28SH2pud'
api_key = 'PBmJ6cIVEU8ctG1CM8ZCIIZLV'
api_secret = 'SUqxW4ml0MR85mH9SybtVgu0p0KUnIhkWSTgEZYoG51JI8APJH'
access_token = '1981392334001754112-f1d947RkSKJtcB238gjM29oN1MeeN5'
access_token_secret = 'CZUQnvqA9jQwitnlQMkUAvtpa5XypLckhaJltKA2NrQjT'


# client = tweepy.Client(
#     bearer_token=bearer_token,
#     wait_on_rate_limit=True  # Enables auto-handling of rate limits
# )

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
        print(f" Error for keyword '{keyword}': {e}")

    df = pd.DataFrame({
        'tweet_id': tweet_id,
        'time': tweet_time,
        'tweet': tweet_text,
        'retweet_count': retweet_count
    })

    filename = f"{keyword.replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f" Saved {len(df)} tweets for '{keyword}' to {filename}")
    return df

#  Keywords for Indian election
keywords = ['BJP', 'Congress']

#  Collect tweets
# for keyword in keywords:
#     get_related_tweets(keyword, max_items=10)
import time
for keyword in keywords:
    get_related_tweets(keyword, max_items=10)
    time.sleep(5)  # 5-second pause between requests

#  Text cleaning function
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[:;\n_1]', '', text)
    return text

# Sentiment analysis function
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

#  Buzz score function
def compute_buzz(row):
    weight = {'positive': 1.5, 'neutral': 1.0, 'negative': 0.5}
    return row['retweet_count'] * weight.get(row['sentiment'], 1.0)

# Combine CSVs
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if csv_files:
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined_df = combined_df[['tweet_id', 'time', 'tweet', 'retweet_count']]
    combined_df.dropna(subset=['tweet_id'], inplace=True)
    combined_df['time'] = combined_df['time'].ffill().bfill()
    combined_df['retweet_count'] = combined_df['retweet_count'].fillna(0)
    combined_df['retweet_count'] = combined_df['retweet_count'].astype(int)
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df.drop_duplicates(subset='tweet_id', keep='last', inplace=True)

    combined_df['tweet'] = combined_df['tweet'].apply(cleanTxt)
    combined_df['sentiment'] = combined_df['tweet'].apply(get_sentiment)
    combined_df['buzz_score'] = combined_df.apply(compute_buzz, axis=1)
    combined_df.to_csv('cleaned_df.csv', index=False)
    print("\n✅ Cleaned dataset with sentiment and buzz score saved as 'cleaned_df.csv'")

else:
    print(" No CSV files found. Skipping merge and cleaning steps.")
    print(" Loading mock data for testing...")

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

#  Sentiment Distribution
print("\n Sentiment Distribution:")
print(combined_df['sentiment'].value_counts())

# 🧾 Sample tweets by sentiment
print("\n Positive Tweets:")
print(combined_df[combined_df['sentiment'] == 'positive'][['time', 'tweet']].head(3))

print("\n Neutral Tweets:")
print(combined_df[combined_df['sentiment'] == 'neutral'][['time', 'tweet']].head(3))

print("\n Negative Tweets:")
print(combined_df[combined_df['sentiment'] == 'negative'][['time', 'tweet']].head(3))

#  Top buzzworthy tweets
print("\n Top Buzz Tweets:")
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
else:
    print("⚠️ No sentiment data available to plot.")



#  Carousel-style summary
positive = sentiment_counts.get('positive', 0)
neutral = sentiment_counts.get('neutral', 0)
negative = sentiment_counts.get('negative', 0)

summary = f"""
🗳️ Election Buzz Summary:
-Positive tweets: {positive}
- Neutral tweets: {neutral}
- Negative tweets: {negative}

Top buzz topics include rallies, speeches, and Election Commission updates.
"""
print(summary)

#  Feature Engineering + ML Prediction
if combined_df.empty:
    print("⚠️ No data available for training. Skipping ML prediction.")
else:
    # ✅ Feature Engineering
    combined_df['tweet_length'] = combined_df['tweet'].apply(len)
    combined_df['has_keywords'] = combined_df['tweet'].apply(lambda x: int('Modi' in x or 'Congress' in x or 'BJP' in x))

    #  Encode sentiment labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    combined_df['sentiment_label'] = le.fit_transform(combined_df['sentiment'])

    # Train-test split
    from sklearn.model_selection import train_test_split
    X = combined_df[['tweet_length', 'has_keywords']]
    y = combined_df['sentiment_label']

    if len(X) < 2:
        print("⚠️ Not enough data for train-test split. Skipping ML prediction.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Train Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        #  Predict and evaluate
        y_pred = model.predict(X_test)
        from sklearn.metrics import classification_report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        #  Show predictions with original tweets
        X_test = X_test.copy()
        X_test['predicted_sentiment'] = le.inverse_transform(y_pred)
        X_test['actual_sentiment'] = le.inverse_transform(y_test)
        X_test['tweet'] = combined_df.loc[X_test.index, 'tweet']
        print("\n🔍 Sample Predictions:")
        print(X_test[['tweet', 'actual_sentiment', 'predicted_sentiment']])
