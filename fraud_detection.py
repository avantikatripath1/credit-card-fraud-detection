import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully")

# Preview dataset
print(df.head())

# Dataset shape
print("Dataset shape:", df.shape)

# Count fraud vs normal transactions
print(df['Class'].value_counts())

# Visualize fraud vs normal transactions
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check dataset info
print(df.info())


scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

# Drop original columns
df.drop(['Time','Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)



smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_resampled.value_counts())

# Logistic Regression
lr_file = "logistic_regression_model.pkl"

if os.path.exists(lr_file):
    print("Loading saved Logistic Regression model...")
    lr = joblib.load(lr_file)
else:
    print("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_resampled, y_train_resampled)
    joblib.dump(lr, lr_file)

lr_predictions = lr.predict(X_test)

lr_predictions = lr.predict(X_test)

# Decision Tree
dt_file = "decision_tree_model.pkl"

if os.path.exists(dt_file):
    print("Loading saved Decision Tree model...")
    dt = joblib.load(dt_file)
else:
    print("Training Decision Tree model...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_resampled, y_train_resampled)
    joblib.dump(dt, dt_file)

dt_predictions = dt.predict(X_test)

# Random Forest
rf_file = "random_forest_model.pkl"

if os.path.exists(rf_file):
    print("Loading saved Random Forest model...")
    rf = joblib.load(rf_file)
else:
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)
    joblib.dump(rf, rf_file)

rf_predictions = rf.predict(X_test)



from sklearn.metrics import classification_report

print("\nLogistic Regression Results")
print(classification_report(y_test, lr_predictions))

print("\nDecision Tree Results")
print(classification_report(y_test, dt_predictions))

print("\nRandom Forest Results")
print(classification_report(y_test, rf_predictions))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, rf_predictions)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = rf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--')  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

import pandas as pd

importances = rf.feature_importances_
features = X.columns

feature_importance = pd.Series(importances, index=features)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
feature_importance.head(10).plot(kind='bar')
plt.title("Top 10 Important Features for Fraud Detection")
plt.ylabel("Importance Score")
plt.show()

