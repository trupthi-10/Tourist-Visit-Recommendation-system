import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("Top Indian Places to Visit.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Encode categorical columns
label_encoders = {}
categorical_cols = ['state', 'significance', 'type']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Create binary classification target: popular (rating >= 4.2)
df['popularity'] = (df['google_review_rating'] >= 4.2).astype(int)

# Define features and target
feature_cols = ['state', 'significance', 'type', 'entrance_fee_in_inr', 'time_needed_to_visit_in_hrs']
X = df[feature_cols]
y = df['popularity']

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# --- MODEL 1: KNN ---
model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

print("📌 Model 1: KNN Classifier")
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred1), 2))
print("🎯 Precision:", round(precision_score(y_test, y_pred1), 2))
print("🔁 Recall:", round(recall_score(y_test, y_pred1), 2))
print("📊 F1 Score:", round(f1_score(y_test, y_pred1), 2))
cm1 = confusion_matrix(y_test, y_pred1)
print("\n🔲 Confusion Matrix:\n", cm1)
ConfusionMatrixDisplay(confusion_matrix=cm1).plot(cmap="Blues")
plt.title("KNN - Confusion Matrix")
plt.show()

# --- MODEL 2: Decision Tree (CART) ---
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

print("\n📌 Model 2: Decision Tree (CART)")
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred2), 2))
print("🎯 Precision:", round(precision_score(y_test, y_pred2), 2))
print("🔁 Recall:", round(recall_score(y_test, y_pred2), 2))
print("📊 F1 Score:", round(f1_score(y_test, y_pred2), 2))
cm2 = confusion_matrix(y_test, y_pred2)
print("\n🔲 Confusion Matrix:\n", cm2)
ConfusionMatrixDisplay(confusion_matrix=cm2).plot(cmap="Greens")
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# --- MODEL 3: Naive Bayes ---
model3 = GaussianNB()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)

print("\n📌 Model 3: Naive Bayes")
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred3), 2))
print("🎯 Precision:", round(precision_score(y_test, y_pred3), 2))
print("🔁 Recall:", round(recall_score(y_test, y_pred3), 2))
print("📊 F1 Score:", round(f1_score(y_test, y_pred3), 2))
cm3 = confusion_matrix(y_test, y_pred3)
print("\n🔲 Confusion Matrix:\n", cm3)
ConfusionMatrixDisplay(confusion_matrix=cm3).plot(cmap="Oranges")
plt.title("Naive Bayes - Confusion Matrix")
plt.show()

# Save all models
joblib.dump(model1, "knn_model.pkl")
joblib.dump(model2, "decision_tree_model.pkl")
joblib.dump(model3, "naive_bayes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
