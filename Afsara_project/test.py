# 📦 Import required libraries
import pandas as pd
import tweepy
import os
import time

# 🔐 Twitter API v2 credentials (replace with your actual values)

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFmX4wEAAAAA%2B6rDoxMk474MEV%2FZaiSmksVWavM%3DmVfVtMMZRBvav3o6Xdv5HhREgHxY89EBiPiBZI7yTJk4ZcADK1'
api_key = 'aDfh5Lx9xCwXWZPOD2Pbrcffr'
api_secret = 'SUqxW4ml0MR85mH9SybtVgu0p0KUnIhkWSTgEZYoG51JI8APJH'
access_token = '1979565204309704704-xZ5ZOz7MVmOinomTVjzl9Nf4QluUO3'
access_token_secret = '7FJzYBoR34BJzIIUKChKWVu35aspJPLfk4j5IbaYiBSHo'

# 🧵 Initialize Tweepy Client for Twitter API v2
client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

# 🐦 Function to collect tweets for a given keyword
def get_related_tweets(keyword, max_items=50):
    tweet_id = []
    tweet_time = []
    tweet_text = []
    retweet_count = []

    try:
        # Search recent tweets (max 100 per request)
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

        # Save to DataFrame and CSV
        df = pd.DataFrame({
            'tweet_id': tweet_id,
            'time': tweet_time,
            'tweet': tweet_text,
            'retweet_count': retweet_count
        })

        filename = f"{keyword.replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} tweets for '{keyword}' to {filename}")
        return df

    except tweepy.TooManyRequests:
        print("⏳ Rate limit hit. Sleeping for 15 minutes...")
        time.sleep(900)
        return get_related_tweets(keyword, max_items)

# 🔍 Keywords for Indian election
keywords = [
    'Indian elections', 'Lok Sabha 2024', 'BJP', 'Congress', 'Narendra Modi',
    'Rahul Gandhi', 'Aam Aadmi Party', 'Arvind Kejriwal', 'INDIA alliance',
    'Election Commission of India', 'vote for change', 'My Vote Matters'
]

# 🔁 Collect tweets for each keyword
for keyword in keywords:
    get_related_tweets(keyword, max_items=50)
    time.sleep(5)  # optional: pause between requests to avoid rate limits

# 📂 Combine all CSVs into one DataFrame
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# 🖨️ Show combined data
print("\n📊 Combined Tweet Data Sample:")
print(combined_df.head())

# 🧼 Clean and filter columns
combined_df = combined_df[['tweet_id','time', 'tweet', 'retweet_count']]

# 🎲 Display a random sample of 20 tweets
print("\n🎯 Sample of 20 Tweets:")
print(combined_df.sample(20))


# checking if there are any missing values in the dataset
combined_df.isnull().sum()


# finding the percentage of missing values for each column
percent_missing = round(combined_df.isnull().sum() * 100 / len(combined_df),2)
percent_missing

# finding the percentage of missing values for the entire dataset
percentage_missing = round(combined_df.isnull().sum().sum() * 100 / len(combined_df), 2)
percentage_missing

combined_df.dropna(subset=['tweet_id'], inplace=True)


# forward filling and backward filling the dates that are missing in our dataset
combined_df['time'] = combined_df['time'].ffill().bfill()


# filling '0' in the missing values in our retweet_count column
combined_df['retweet_count'] = combined_df['retweet_count'].fillna(0)


# checking to see if the missing values have been imputed
combined_df.isnull().sum()


# checking the datatypes of each column
combined_df.info()

# changing the datatype of retweet_count and time to their respective datatypes
combined_df['retweet_count'] = combined_df['retweet_count'].astype('int')
combined_df['time'] = pd.to_datetime(combined_df['time'])
combined_df.info()


# checking if we have duplicates in our 'tweet_id' column since it is out unique identifier
combined_df['tweet_id'].duplicated().sum()


# dropping the duplicates in the tweet_id column
combined_df.drop_duplicates(subset=['tweet_id'], keep='last', inplace=True)


# Number of unique values in the elections dataset
print('Cardinality of the dataset columns:\n')
for i in combined_df.columns:
  print(f'{i:<5} : {combined_df[i].nunique()}')

  # checking the shape of our final dataset
  combined_df.shape

  # saving the cleaned dataset to a csv file
  combined_df.to_csv('cleaned_df.csv')

  #EDA

  clean_df = pd.read_csv('cleaned_df.csv')

  # Creating a Function to clean the tweets further

  import re


  def cleanTxt(text):
      text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))  # Removing @mentions
      text = re.sub(r'#', '', str(text))  # Removing the '#' symbol
      text = re.sub(r'RT[\s]+', '', str(text))  # Removing RT
      text = re.sub(r'https?:\/\/\S+', '', str(text))  # Removing the hyper link
      text = re.sub(r':', '', str(text))  # Removing the ':'
      text = re.sub(r';', '', str(text))  # Removing the ';'
      text = re.sub(r'\n', '', str(text))  # Removing the '\n'
      text = re.sub(r'1', '', str(text))  # Removing the '1'
      text = re.sub(r'_', '', str(text))  # Removing the '1'

      return text


  clean_df['tweet'] = clean_df['tweet'].apply(cleanTxt)

  # Showing cleaned text

  clean_df.head()

  clean_df.shape