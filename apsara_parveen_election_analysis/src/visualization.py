import matplotlib.pyplot as plt


def summarize_sentiment(df):
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts())

    sentiment_counts = df['sentiment'].value_counts()
    if not sentiment_counts.empty:
        sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.tight_layout()
        plt.show()
