# Neural-Network-SMS-Text-Classifier

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('sms_spam_collection.csv', delimiter='\t', header=None, names=['label', 'message'])

# Display the first few rows of the dataset
data.head()


# Convert labels to binary: ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into features and target variable
X = data['message']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that first vectorizes the text data and then applies the Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)


def predict_message(message):
    # Predict the class (0 for ham, 1 for spam) and the probability
    prediction_proba = model.predict_proba([message])[0]
    prediction = model.predict([message])[0]
    
    # Get the likelihood of "ham" or "spam"
    likelihood = prediction_proba[0] if prediction == 0 else prediction_proba[1]
    
    # Map the numeric prediction back to the string labels
    label = 'ham' if prediction == 0 else 'spam'
    
    return [likelihood, label]


# Test the model on the test dataset
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Example of testing the prediction function
print(predict_message("Congratulations! You've won a $1,000 gift card. Call now!"))
print(predict_message("Hey, are we still on for lunch today?"))
