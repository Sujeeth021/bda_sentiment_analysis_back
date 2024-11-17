import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load and preprocess the datasets
# training_file_path = "C:\\Users\\sk\\Downloads\\archive (1)\\twitter_training.csv"  # Adjust this path
# validation_file_path = "C:\\Users\\sk\\Downloads\\archive (1)\\twitter_validation.csv"  # Adjust this path

training_file_path = "twitter_training.csv"
validation_file_path = "twitter_validation.csv"


training_data = pd.read_csv(training_file_path)
training_data.columns = ['ID', 'Source', 'Sentiment', 'Tweet']
training_data = training_data[['Sentiment', 'Tweet']].dropna(subset=['Tweet'])

# Encode sentiment labels
label_encoder = LabelEncoder()
training_data['Sentiment'] = label_encoder.fit_transform(training_data['Sentiment'])

# Vectorize tweet text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(training_data['Tweet'])
y_train = training_data['Sentiment']

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Define API route for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Transform input text
    text_vectorized = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(text_vectorized)
    sentiment = label_encoder.inverse_transform(prediction)[0]

    return jsonify({'sentiment': sentiment})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
