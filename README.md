# 👥 Employee Attrition Prediction

An end-to-end **Machine Learning application** that predicts whether an employee is likely to **leave or stay with a company** based on demographic, compensation, and workplace-related attributes.

The project implements a complete ML workflow — from data preprocessing and feature scaling to model training, evaluation, serialization, and real-time prediction through an interactive **Streamlit** web application.

---

## 📖 Project Overview

Employee attrition can have a significant impact on workforce stability and organizational costs. This project uses **Logistic Regression**, a supervised machine learning algorithm, to identify patterns in employee data and predict attrition.

The trained model uses eight employee attributes:

* Age
* Distance from Home
* Monthly Income
* Years at Company
* Job Satisfaction
* Overtime
* Number of Companies Worked
* Training Hours

Categorical `Overtime` values (`Yes` / `No`) are converted into numerical values before training and prediction.

---

## ✨ Key Features

* 🤖 **Logistic Regression Classifier** for binary employee attrition prediction
* 📊 **Feature Preprocessing** for numerical and categorical attributes
* 📏 **StandardScaler** for feature standardization
* 🎯 **Stratified Train-Test Split** to maintain class distribution
* 📈 **Model Evaluation** using accuracy
* 💾 **Model Serialization** using Joblib
* 🌐 **Interactive Streamlit Interface** for real-time predictions
* ⚡ **Pre-trained Model Inference** without retraining when the application runs

---

## 🔄 Machine Learning Workflow

```text
Employee Dataset
       │
       ▼
Data Preprocessing
       │
       ├── Overtime Encoding
       │
       ▼
Feature Selection
       │
       ▼
Stratified Train/Test Split
       │
       ▼
StandardScaler
       │
       ▼
Logistic Regression
       │
       ▼
Model Evaluation
       │
       ▼
Joblib Serialization
       │
       ├── Employee_Attrition_Model.pkl
       └── Employee_Attrition_Scaler.pkl
       │
       ▼
Streamlit Application
       │
       ▼
Real-Time Attrition Prediction
```

The training pipeline uses an 80/20 train-test split with stratification, scales the features using `StandardScaler`, and trains a `LogisticRegression` classifier.

---

## 🛠️ Tech Stack

| Technology       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| **Python**       | Core programming language                  |
| **Pandas**       | Data loading and manipulation              |
| **NumPy**        | Numerical computations                     |
| **Scikit-learn** | Preprocessing, model training & evaluation |
| **Joblib**       | Model and scaler serialization             |
| **Streamlit**    | Interactive web application                |

---

## 📂 Project Structure

```text
attrition_ml_project/
│
├── app.py                              # Streamlit prediction application
├── Train_model.py                      # Data preprocessing & model training
├── data.csv                            # Employee dataset
├── requirements.txt                    # Python dependencies
│
├── Employee_Attrition_Model.pkl        # Trained Logistic Regression model
├── Employee_Attrition_Scaler.pkl      # Fitted StandardScaler
│
├── screenshots/
│   ├── app.png                         # Application interface in SS folder
│   ├── input.png                       # Employee input interface in SS folder
│   └── prediction.png                  # Prediction result in SS folder
│
└── .gitignore
```

---

## 🖥️ Streamlit Application

The application provides an interactive interface where users can enter employee information including age, distance from home, monthly income, years at the company, job satisfaction, overtime, previous companies worked, and training hours.

The application then:

1. Converts the `Overtime` input into a numerical value.
2. Creates a DataFrame containing the employee's attributes.
3. Applies the previously fitted scaler.
4. Passes the transformed data to the trained Logistic Regression model.
5. Displays whether the employee is predicted to leave or stay.

---

## 📊 Model Training & Evaluation

The model is trained using `LogisticRegression` after standardizing the training features with `StandardScaler`.

The test set is used to generate predictions, after which the model's **accuracy** is calculated using `accuracy_score`.

```python
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.4f}')
```

The trained model and fitted scaler are then serialized using Joblib for use by the Streamlit application.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd attrition_ml_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python Train_model.py
```

This creates:

```text
Employee_Attrition_Model.pkl
Employee_Attrition_Scaler.pkl
```

### 4. Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 🎯 Project Highlights

This project demonstrates practical implementation of:

* Supervised machine learning
* Binary classification
* Data preprocessing
* Categorical feature encoding
* Feature standardization
* Stratified train-test splitting
* Logistic Regression
* Model evaluation
* Model serialization
* Real-time ML inference
* Streamlit application development

---

## 📄 License

This project is intended for educational and demonstration purposes.
