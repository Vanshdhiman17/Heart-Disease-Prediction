# ❤️ Heart Disease Prediction Using Machine Learning

A machine learning classification project that predicts the likelihood of heart disease based on patient health and clinical information.

The project includes **data analysis, preprocessing, model training, model comparison, and an interactive Streamlit web application**.

---

## 📌 Features

* Exploratory Data Analysis
* Data cleaning and preprocessing
* Categorical feature encoding
* Feature scaling
* Multiple machine learning models
* Model comparison using Accuracy and F1 Score
* Automatic selection of the best model
* Interactive Streamlit prediction application
* Saved trained model using Joblib

---

## 🔄 Workflow

```text
📊 Dataset
    →
🔍 Data Analysis
    →
🧹 Data Cleaning
    →
🔤 Feature Encoding
    →
✂️ Train-Test Split
    →
⚖️ Feature Scaling
    →
🤖 Model Training
    →
📈 Model Evaluation
    →
🏆 Best Model Selection
    →
💾 Model Saving
    →
🌐 Streamlit App
    →
👤 User Input
    →
🔮 Prediction
```

---

## 🤖 Machine Learning Models

The following algorithms are trained and compared:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Gaussian Naive Bayes
* Support Vector Machine (SVM)

The model with the highest **F1 Score** is automatically selected as the final model.

---

## 📊 Dataset

The project uses a `heart.csv` dataset containing patient health and clinical information.

### Main Features

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

### Target

`HeartDisease`

* `0` → No Heart Disease
* `1` → Heart Disease

---

## 🧹 Data Preprocessing

The project performs the following preprocessing steps:

* Replaces invalid `Cholesterol = 0` values with the mean of valid values
* Converts categorical features using one-hot encoding
* Splits the dataset into 80% training and 20% testing data
* Uses stratified train-test splitting
* Scales features using `StandardScaler`

---

## 📈 Model Evaluation

The models are evaluated using:

* **Accuracy**
* **F1 Score**

### Model Performance

| Model                        | Accuracy | F1 Score |
| ---------------------------- | -------: | -------: |
| Logistic Regression          | `89%` | `90%` |
| K-Nearest Neighbors (KNN)    | `88%` | `89%` |
| Decision Tree                | `74%` | `76%` |
| Gaussian Naive Bayes         | `87%` | `88%` |
| Support Vector Machine (SVM) | `85%` | `87%` |

The model with the highest **F1 Score** is selected and saved for use in the Streamlit application.

---

## 📤 Output

The trained model is integrated into an interactive Streamlit application.

Users enter patient information and receive a prediction from the trained model.

### High Risk

```text
⚠️ High Risk of Heart Disease
```

### Low Risk

```text
✅ Low Risk of Heart Disease
```
---

## 📊 Visualizations

### Heart Disease Distribution

Heart Disease Distribution.png

### Feature Distributions

Feature_Distributions.png

### Chest Pain Type vs Heart Disease

Chest Pain Type vs Heart Disease.png

### Correlation Heatmap

Correlation_Heatmap.png

## 🌐 Streamlit Application

The application allows users to enter:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

The entered information is processed using the same feature structure and scaler used during model training before being passed to the trained model.

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python heart_disease_prediction.py
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Model explainability
* Improved Streamlit UI
* Prediction probability
* Docker deployment
* Online deployment
* FastAPI backend

---

## 👨‍💻 Author

**Vansh Dhiman**

⭐ If you found this project useful, consider starring the repository.
