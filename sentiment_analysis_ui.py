import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import gradio as gr

# Load dataset
data = pd.read_csv("customer_reviews.csv")
X = data['review']
y = data['sentiment']

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X_vec = tfidf.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Function for prediction
def predict_sentiment(text):
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    return f"Predicted Sentiment: {pred.capitalize()}"

# Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your review here...", label="Review"),
    outputs=gr.Textbox(label="Prediction"),
    title="Customer Review Sentiment Analyzer",
    description="Enter a review to predict if sentiment is Positive, Neutral, or Negative.",
    theme="default"
)

interface.launch()
