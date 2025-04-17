import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import numpy as np

# Custom transformer to extract a single column
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column]

def load_and_prepare_data(fake_path, true_path):
    # Load datasets
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake['label'] = 0  # 0 for fake
    true['label'] = 1  # 1 for real
    data = pd.concat([fake, true], ignore_index=True)
    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def build_pipeline():
    # Combine title and text features using TF-IDF
    feature_union = FeatureUnion([
        ('title_tfidf', Pipeline([
            ('extract', ColumnExtractor('title')),
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000))
        ])),
        ('text_tfidf', Pipeline([
            ('extract', ColumnExtractor('text')),
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=7000))
        ])),
        # Optionally add subject
        # ('subject_tfidf', Pipeline([
        #     ('extract', ColumnExtractor('subject')),
        #     ('tfidf', TfidfVectorizer(stop_words='english', max_features=100))
        # ])),
    ])
    pipeline = Pipeline([
        ('features', feature_union),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipeline

import os

def main():
    fake_path = 'Fake.csv'
    true_path = 'True.csv'

    # Check if files exist in the current directory
    if not os.path.isfile(fake_path):
        print(f"Error: Could not find {fake_path} in the current directory: {os.getcwd()}")
        print("Please make sure the file is present and the name matches exactly, including case.")
        return
    if not os.path.isfile(true_path):
        print(f"Error: Could not find {true_path} in the current directory: {os.getcwd()}")
        print("Please make sure the file is present and the name matches exactly, including case.")
        return

    data = load_and_prepare_data(fake_path, true_path)

    X = data[['title', 'text']]  # Add 'subject' if you want
    y = data['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predict probabilities
    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    # Example: Show probability output for first 10 samples
    for i in range(10):
        print(f"Title: {X_test.iloc[i]['title']}")
        print(f"Predicted: {'Real' if y_pred[i] else 'Fake'} (Probability Real: {y_proba[i][1]*100:.1f}%)\n")

if __name__ == '__main__':
    main()
