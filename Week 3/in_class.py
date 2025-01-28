import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import csv

# read csv file emails.csv
with open('emails.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Convert to DataFrame
df = pd.DataFrame(data[1:], columns=data[0])
df = df.dropna()
X = df['Message']
y = df['Label']

# Convert text to numerical features
vectorizer = CountVectorizer()
X_transformed = vectorizer.fit_transform(X)
X_transformed = X_transformed.toarray()


# Separate features and labels for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
