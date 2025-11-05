# 📦 Import required libraries
import tweepy
import pandas as pd
import time

# 🔐 Twitter API v2 credentials (replace with your actual values)
# 🔐 Twitter API v2 credentials (replace with your actual values)
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFmX4wEAAAAA%2B6rDoxMk474MEV%2FZaiSmksVWavM%3DmVfVtMMZRBvav3o6Xdv5HhREgHxY89EBiPiBZI7yTJk4ZcADK1'
api_key = 'aDfh5Lx9xCwXWZPOD2Pbrcffr'
api_secret = 'SUqxW4ml0MR85mH9SybtVgu0p0KUnIhkWSTgEZYoG51JI8APJH'
access_token = '1979565204309704704-xZ5ZOz7MVmOinomTVjzl9Nf4QluUO3'
access_token_secret = '7FJzYBoR34BJzIIUKChKWVu35aspJPLfk4j5IbaYiBSHo'
# ✅ Initialize Tweepy Client for API v2
client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

# 🐦 Function to collect tweets using API v2
def get_indian_election_tweets(keyword, max_items=10):  #️⃣ Reduced max_items to avoid rate limits
    tweet_id = []
    tweet_time = []
    tweet_text = []
    retweet_count = []
    keyword_tag = []

    print(f"🔍 Collecting tweets for: {keyword}")
    try:
        for tweet in tweepy.Paginator(
            client.search_recent_tweets,
            query=f"{keyword} lang:en -is:retweet",  #️⃣ Filters for English tweets and excludes retweets
            tweet_fields=['id', 'created_at', 'public_metrics', 'text'],
            max_results=10
        ).flatten(limit=max_items):  #️⃣ Limits total tweets per keyword
            tweet_id.append(tweet.id)
            tweet_time.append(tweet.created_at)
            tweet_text.append(tweet.text)
            retweet_count.append(tweet.public_metrics['retweet_count'])
            keyword_tag.append(keyword)

    except tweepy.TooManyRequests:  #️⃣ Graceful handling of rate limits
        print(f"⚠️ Rate limit hit for '{keyword}'. Skipping.")
        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error for keyword '{keyword}': {e}")
        return pd.DataFrame()

    df = pd.DataFrame({
        'tweet_id': tweet_id,
        'time': tweet_time,
        'tweet': tweet_text,
        'retweet_count': retweet_count,
        'keyword': keyword_tag
    })

    return df  #️⃣ No per-keyword CSV — only one combined file

# 🔍 Keywords related to Indian elections
keywords = ['Indian elections']


# 🔁 Collect and combine all tweets
all_dfs = []
for kw in keywords:
    df_kw = get_indian_election_tweets(kw, max_items=10)  #️⃣ Reduced tweet count per keyword
    if not df_kw.empty:  #️⃣ Skip empty results
        all_dfs.append(df_kw)
    time.sleep(10)  #️⃣ Pause to respect rate limits

# 📁 Save one combined dataset
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv('indian_election_tweets.csv', index=False)  #️⃣ One single CSV file
    print(f"\n✅ Saved {len(combined_df)} tweets to 'indian_election_tweets.csv'")
else:
    print("⚠️ No tweets collected. Check keywords or API access.")