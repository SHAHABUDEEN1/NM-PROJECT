# Traffic-Accident-Analysis-and-Prediction
# Enhancing Road Safety with AI-driven Traffic Accident Analysis and Prediction
# Author: SHAHABUDEEN NS
# Date: 03.05.2025
# Description: This project analyzes traffic accident data and builds a predictive model to classify accident severity.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('traffic_accidents.csv')  # Example file name

# Preview data
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())


# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
categorical_cols = ['Weather_Condition', 'Road_Surface', 'Light_Condition']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Feature Scaling
numerical_cols = ['Speed_Limit', 'Vehicle_Age', 'Driver_Age']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# Accident severity distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Accident_Severity', data=df)
plt.title('Accident Severity Distribution')
plt.xlabel('Severity')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Define features and target
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
