# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: CHUKKA JOSHNA PRIYA

INTERN ID: CT04DN841

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

##

During my four-week internship at CodTech IT Solutions under the guidance of mentor Neela Santosh, I worked on a project titled "Customer Review Sentiment Analysis" using logistic regression and a real-time interface built with Gradio. The main objective was to develop a machine learning model that classifies customer reviews as Positive, Neutral, or Negative, and deploy it using an interactive UI for live prediction.

The entire solution was implemented in Python using libraries such as pandas, scikit-learn, and gradio.

1. Dataset Loading and Preprocessing
The project began with importing and reading the customer reviews dataset stored in a CSV file. The dataset consisted of two primary columns: review (text data) and sentiment (labels). The review column was extracted as the feature (X), and sentiment was used as the target (y).

2. Text Vectorization
Text data needed to be converted into numerical format for machine learning. This was accomplished using TF-IDF vectorization (TfidfVectorizer from scikit-learn). The vectorizer removed English stopwords and transformed the entire review text corpus into a sparse matrix of TF-IDF features.

3. Model Training
After preprocessing, the data was split into training and testing sets using an 80-20 ratio with train_test_split(). A Logistic Regression classifier was selected for its efficiency in binary and multi-class text classification tasks. The model was trained on the TF-IDF-transformed training data.

4. Evaluation
Once trained, the model was tested on the unseen test data. The accuracy score and a complete classification report (including precision, recall, and F1-score for each sentiment class) were generated using accuracy_score() and classification_report() from sklearn.metrics.

5. Live Prediction Interface
To allow real-time sentiment prediction for any review text, a frontend interface was created using the Gradio library. A function predict_sentiment() was defined, which accepts a string input, vectorizes it using the trained TF-IDF model, and returns the predicted sentiment label using the trained logistic regression model.

6. Gradio UI Integration
The Gradio interface was built with a clean layout:

Input Box: Allows users to enter any customer review.

Output Box: Displays the predicted sentiment.

Title and Description: Provides a summary of the system and its functionality.

With a single command (interface.launch()), the entire system was deployed in the browser, making it easy to test and use without any complex setup.

##

#OUTPUT
