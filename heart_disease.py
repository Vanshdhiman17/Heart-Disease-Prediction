import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("heart.csv")

print(df.head())
print(df.shape)
print(df.describe())
print(df.info())


# Target distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="HeartDisease", data=df)
plt.title("Heart Disease Distribution")
plt.savefig("Heart Disease Distribution.png")
plt.show()

# Feature distributions
plt.figure(figsize=(10, 8))
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)
    plt.title(var)

plotting("Age", 1)
plotting("RestingBP", 2)
plotting("Cholesterol", 3)
plotting("MaxHR", 4)

plt.tight_layout()
plt.savefig("Feature Distributions.png", dpi=300, bbox_inches="tight")
plt.show()


# Replace 0 cholesterol values with mean
ch_mean = df.loc[
    df["Cholesterol"] != 0,
    "Cholesterol"
].mean()

df["Cholesterol"] = df["Cholesterol"].replace(0, ch_mean)


# Chest pain analysis
plt.figure(figsize=(6, 5))
sns.countplot(x="ChestPainType",hue="HeartDisease",data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.savefig("Chest Pain Type vs Heart Disease.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title("Correlation Heatmap")
plt.savefig("Correlation Heatmap.png")
plt.show()

# Convert categorical variables into numerical variables
df = pd.get_dummies(df,drop_first=True,dtype=int)
print(df.head())

# Separate features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "F1 Score": round(f1, 4)
    })


# Display results
results_df = pd.DataFrame(results)

print("\nModel Performance:")
print(results_df.to_string(index=False))

# Find best model based on F1 score
best_model_name = results_df.loc[results_df["F1 Score"].idxmax(),"Model"]
best_model = models[best_model_name]
print("\nBest Model:", best_model_name)

# Save model, scaler and column names
joblib.dump(best_model,"heart_disease_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(X.columns.tolist(),"columns.pkl")
 
print("\nModel saved successfully!")