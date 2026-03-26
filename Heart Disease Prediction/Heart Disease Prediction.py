#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

#Load dataset
df = pd.read_csv("heart.csv")

#Target distribution
sns.countplot(x="target", data=df)
plt.title("Heart Disease Presence (0 = No, 1 = Yes)")
plt.grid(True)
plt.show()

#Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

#Split features and target
X = df.drop("target", axis=1)
y = df["target"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Test predictions
y_pred = model.predict(X_test)

#Evaluation results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#ROC Curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

#Sample patient data
sample_patient = {
    "age": 52,
    "sex": 1,
    "cp": 3,
    "trestbps": 110,
    "chol": 150,
    "fbs": 0,
    "restecg": 1,
    "thalach": 160,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 2
}

#Prepare input for prediction
user_df = pd.DataFrame([sample_patient])

#Make prediction
prediction = model.predict(user_df)

#Display result
if prediction[0] == 1:
    print("\nResult: Heart Disease.")
else:
    print("\nResult: No Heart Disease.")
