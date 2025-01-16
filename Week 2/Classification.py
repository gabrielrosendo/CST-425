import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns

data = sns.load_dataset('titanic')  
print(data.head())

# Check basic information about the dataset
print(data.info())  

# One-hot encoding
data = pd.get_dummies(data, columns=['sex', 'embarked'], drop_first=True)

print(data.head())  

# Drop rows with Null
data = data.dropna()

# Drop non numeric data
X = data.drop(columns=['survived', 'class', 'embark_town', 'alive','who','deck']) 
y = data['survived']
print(f"size", len(X))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize  model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report (precision, recall, F1-score)
print(classification_report(y_test, y_pred))