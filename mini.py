# Loan Approval Prediction using Decision Tree and Random Forest
# Works in Google Colab, Jupyter or any Python setup.
# Dataset: Kaggle "Loan Prediction Problem Dataset" (train CSV).
# Save it as "loan_data.csv" in the same folder (in Colab: upload it first).

# ---------------------------------------------------------
# STEP 1: Import libraries
# ---------------------------------------------------------
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# ---------------------------------------------------------
# STEP 2: Load the data
# ---------------------------------------------------------
df = pd.read_csv("loan_data.csv")
print("Shape (rows, columns):", df.shape)
print(df.head())

# Loan_ID is just a label, it does not help prediction
df = df.drop("Loan_ID", axis=1)

# ---------------------------------------------------------
# STEP 3: Clean missing values
# Text columns -> most common value, number columns -> median
# ---------------------------------------------------------
print("\nMissing values before cleaning:\n", df.isnull().sum())

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nTotal missing values after cleaning:", df.isnull().sum().sum())

# ---------------------------------------------------------
# STEP 4: Convert text to numbers (models only understand numbers)
# Example: Male/Female -> 1/0, Y/N -> 1/0
# ---------------------------------------------------------
for col in df.select_dtypes(exclude="number").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# ---------------------------------------------------------
# STEP 5: Split into inputs (X) and output (y), then train/test
# 80% to teach the model, 20% to test it on unseen data
# ---------------------------------------------------------
X = df.drop("Loan_Status", axis=1)   # income, credit history, etc.
y = df["Loan_Status"]                # 1 = Approved, 0 = Rejected

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining rows:", len(X_train), "| Testing rows:", len(X_test))

# ---------------------------------------------------------
# STEP 6: Train the Decision Tree
# max_depth=4 keeps the tree small so it is easy to draw and explain
# ---------------------------------------------------------
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# ---------------------------------------------------------
# STEP 7: Train the Random Forest (100 trees voting together)
# ---------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# ---------------------------------------------------------
# STEP 8: Compare results
# ---------------------------------------------------------
print("\n===== RESULTS =====")
print(f"Decision Tree accuracy : {dt_acc * 100:.2f}%")
print(f"Random Forest accuracy : {rf_acc * 100:.2f}%")

print("\nDecision Tree report:\n",
      classification_report(y_test, dt_pred, target_names=["Rejected", "Approved"]))
print("Random Forest report:\n",
      classification_report(y_test, rf_pred, target_names=["Rejected", "Approved"]))

# ---------------------------------------------------------
# STEP 9: Visualisations (use these screenshots in your slides)
# ---------------------------------------------------------

# 9a. Accuracy comparison bar chart
plt.figure(figsize=(5, 4))
bars = plt.bar(["Decision Tree", "Random Forest"], [dt_acc * 100, rf_acc * 100],
               color=["#4C9BE8", "#2ECC71"])
plt.bar_label(bars, fmt="%.1f%%")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.show()

# 9b. The Decision Tree drawn as a flowchart
plt.figure(figsize=(16, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=["Rejected", "Approved"],
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree (depth = 4)")
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()

# 9c. Confusion matrices (correct vs wrong predictions)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, dt_pred, display_labels=["Rejected", "Approved"], ax=axes[0], colorbar=False)
axes[0].set_title("Decision Tree")
ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred, display_labels=["Rejected", "Approved"], ax=axes[1], colorbar=False)
axes[1].set_title("Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# 9d. Feature importance: which input matters most?
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
importance.plot(kind="barh", color="#F39C12")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nTop 3 important features:")
print(importance.sort_values(ascending=False).head(3))

# ---------------------------------------------------------
# STEP 10: Predict for a new applicant (demo for your presentation)
# We reuse one row from the test set so the format is already correct.
# ---------------------------------------------------------
sample = X_test.iloc[[0]]
result = rf_model.predict(sample)[0]
print("\nSample applicant details:\n", sample.T)
print("Prediction:", "Loan APPROVED" if result == 1 else "Loan REJECTED")