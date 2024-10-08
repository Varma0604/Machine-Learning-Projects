import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age', 'outcome']
df = pd.read_csv(url, names=column_names)

# Display basic information about the dataset
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Replace 0 values with NaN for certain columns
zero_columns = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
df[zero_columns] = df[zero_columns].replace(0, np.NaN)

# Impute missing values with mean
for column in zero_columns:
    df[column].fillna(df[column].mean(), inplace=True)

# Split the data into features (X) and target (y)
X = df.drop('outcome', axis=1)
y = df['outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Diabetes Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("\nVisualization images (feature_importance.png and confusion_matrix.png) have been saved.")

# Function to predict diabetes for new data
def predict_diabetes(new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = rf_classifier.predict(new_data_scaled)
    probability = rf_classifier.predict_proba(new_data_scaled)[:, 1]
    return prediction[0], probability[0]

# Example usage of the prediction function
new_person = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
prediction, probability = predict_diabetes(new_person)
print(f"\nPrediction for new person: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
print(f"Probability of having diabetes: {probability:.2f}")