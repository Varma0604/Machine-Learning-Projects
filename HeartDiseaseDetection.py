# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importing models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Load the dataset
df = pd.read_csv('/kaggle/input/heart-failure-prediction/heart.csv')

# Data exploration (optional)
print(df.head())
print(df.info())
print(df.describe())

# Visualizing the distributions of numerical columns with boxplots
columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
for column in columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=column)
    plt.title(column)
    plt.show()

# Removing rows with zero values for RestingBP and Cholesterol
df = df[(df['RestingBP'] != 0) & (df['Cholesterol'] != 0)]

# Encoding categorical variables
encoder = LabelEncoder()
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Feature scaling (StandardScaler)
scaler = StandardScaler()
columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Splitting data into features (X) and target (y)
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Function to evaluate models
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print(f"Training Accuracy: {model.score(X_train, y_train)}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
evaluate_model(lr_model)

# Support Vector Classifier
svc = SVC(C=15, gamma=0.001)
evaluate_model(svc)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, min_impurity_decrease=0.001)
evaluate_model(rf_model)

# Bagging Classifier (with RandomForest)
bagg = BaggingClassifier(estimator=RandomForestClassifier(n_estimators=200, min_impurity_decrease=0.001), n_estimators=100)
evaluate_model(bagg)

# Extra Trees Classifier
ex_model = ExtraTreesClassifier(n_estimators=100, min_impurity_decrease=0.001)
evaluate_model(ex_model)

# AdaBoost Classifier
ada = AdaBoostClassifier(estimator=RandomForestClassifier(n_estimators=300, min_impurity_decrease=0.001), n_estimators=30)
evaluate_model(ada)

# XGBoost
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.001, verbosity=0)
evaluate_model(xgb)

# CatBoost
cat = CatBoostClassifier(verbose=0)
evaluate_model(cat)

# LightGBM
lgbm = LGBMClassifier()
evaluate_model(lgbm)

# Feature importance visualization (for RandomForest)
plt.figure(figsize=(10, 6))
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances (RandomForest)')
plt.show()

# Hyperparameter Tuning Example (RandomForest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'min_impurity_decrease': [0.001, 0.0001, 0],
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Best RandomForest model after tuning
print(f"Best Parameters: {grid_rf.best_params_}")
best_rf = grid_rf.best_estimator_
evaluate_model(best_rf)
