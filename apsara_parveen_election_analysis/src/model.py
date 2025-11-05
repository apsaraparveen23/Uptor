from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd


def train_model(df):
    from .features import add_features

    df = add_features(df)
    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])

    X = df[['tweet', 'tweet_length', 'has_keywords', 'num_words', 'contains_link', 'polarity', 'pos_word_count', 'neg_word_count']]
    y = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(max_features=100), 'tweet'),
        ('num', 'passthrough', ['tweet_length', 'has_keywords', 'num_words', 'contains_link', 'polarity', 'pos_word_count', 'neg_word_count'])
    ])

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

    sample = X_test.copy()
    sample['predicted_sentiment'] = le.inverse_transform(y_pred)
    sample['actual_sentiment'] = le.inverse_transform(y_test)
    print("\nSample Predictions:")
    print(sample[['tweet', 'actual_sentiment', 'predicted_sentiment']].head(10))

    return pipeline, le
