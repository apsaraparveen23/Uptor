import os
import pandas as pd
import joblib
from .Apsara_preprocessing import load_manual_csv
from .Apsara_sentiment import get_sentiment, compute_buzz
from .Apsara_visualization import summarize_sentiment
from .Apsara_model import train_model
from .Apsara_features import add_features



def run_pipeline():
    cleaned_path = 'data/cleaned_df.csv'

    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    else:
        df = load_manual_csv()

    df['tweet'] = df['tweet'].apply(str)
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    df['buzz_score'] = df.apply(compute_buzz, axis=1)
    summarize_sentiment(df)

    model, le = train_model(df)

    joblib.dump(model, 'model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("Model and label encoder saved successfully!")

    # Predict new tweets
    new_tweets = pd.DataFrame({
        'tweet': [
            "PM Modi’s new initiative to support small farmers is receiving widespread appreciation across states.",
            "Congress spokesperson questioned the transparency of the newly launched economic policy.",
            "Citizens are hopeful that the government’s renewable energy plan will reduce pollution and create jobs",
            "Opposition parties slammed the administration for the delay in implementing welfare schemes.",
            "Congress launches campaign focused on rural development.",
            "BJP celebrates successful completion of highway project.",
            "Modi's silence on recent protests sparks debate online."
        ]
    })

    new_tweets = add_features(new_tweets)
    predictions = model.predict(new_tweets[['tweet', 'tweet_length', 'has_keywords', 'num_words', 'contains_link', 'polarity', 'pos_word_count', 'neg_word_count']])
    new_tweets['predicted_sentiment'] = le.inverse_transform(predictions)
    new_tweets.to_csv('outputs/new_tweet_predictions.csv', index=False)
    print("\nPredictions saved to outputs/new_tweet_predictions.csv")
