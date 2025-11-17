from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Import your helper functions
from src.Apsara_features import add_features
from src.Apsara_preprocessing import clean_text
from src.Apsara_sentiment import get_sentiment, compute_buzz
from src.Apsara_pipeline import run_pipeline
from src.Apsara_model import train_model

app = Flask(__name__)

# ======================
# 🔹 Load model & encoder
# ======================
MODEL_PATH = "model.pkl"
ENCODER_PATH = "label_encoder.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Loaded saved model and label encoder.")
else:
    print("⚠️ Model not found — training a new one...")
    df = pd.read_csv('data/cleaned_df.csv') if os.path.exists('data/cleaned_df.csv') else run_pipeline()
    model, label_encoder = train_model(df)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("✅ Model trained and saved.")


# ======================
# 🔹 Route 1: Run pipeline
# ======================
@app.route('/run_pipeline', methods=['GET'])
def run_full_pipeline():
    """Runs the full data + sentiment analysis pipeline"""
    try:
        run_pipeline()
        return jsonify({"message": "Pipeline executed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================
# 🔹 Route 2: Predict
# ======================
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Predict sentiment for new tweets"""
    try:
        data = request.get_json()
        tweets = data.get("tweets", [])

        if not tweets:
            return jsonify({"error": "No tweets provided"}), 400

        # Convert to DataFrame
        new_tweets_df = pd.DataFrame({'tweet': tweets})

        # Clean and add features
        new_tweets_df['tweet'] = new_tweets_df['tweet'].apply(clean_text)
        new_tweets_df = add_features(new_tweets_df)

        # Select the same columns used during training
        feature_cols = [
            'tweet', 'tweet_length', 'has_keywords', 'num_words',
            'contains_link', 'polarity', 'pos_word_count', 'neg_word_count'
        ]

        predicted = model.predict(new_tweets_df[feature_cols])
        sentiments = label_encoder.inverse_transform(predicted)
        new_tweets_df['predicted_sentiment'] = sentiments

        return jsonify({
            "predictions": new_tweets_df[['tweet', 'predicted_sentiment']].to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================
# 🔹 Run server
# ======================
if __name__ == "__main__":
    app.run(debug=True)
