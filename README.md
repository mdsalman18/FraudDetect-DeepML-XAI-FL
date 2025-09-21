To generate a complete README for your Flask-based **Fraud Detection System**, here’s a detailed version that explains setup, features, and usage:

---

# 🕵️ Fraud Detection System (Flask + ML)

This project is a full-stack web application designed to detect fraudulent financial transactions using machine learning and deep learning models. It includes both a **remote user portal** and an **admin dashboard**, integrated with various classification models (DNN, SVM, Logistic Regression, Gradient Boosting).

---

## 📁 Project Structure

```
fraud-detection/
│
├── static/                   # Static files (CSS, JS, etc.)
├── templates/                # HTML templates
│
├── updated_synthetic_fraud_dataset.csv
├── model_results.json
├── fraud_detection_dnn_model.h5
├── svm_fraud_model.pkl
├── logistic_regression_fraud_model.pkl
├── gradient_boosting_fraud_model.pkl
│
├── app.py                    # Main Flask app
└── README.md
```

---

## ⚙️ Features

* 🔐 **User Authentication**

  * Remote User: Registration/Login
  * Service Provider (Admin): Login (hardcoded)

* 📊 **ML Models**

  * Deep Neural Network
  * Support Vector Machine
  * Logistic Regression
  * Gradient Boosting Classifier

* 📦 **Functionality**

  * Transaction prediction using trained models
  * Dashboard to compare model accuracy
  * Store & visualize results
  * View user profiles and predictions

---

## 🛠️ Technologies Used

* **Backend**: Flask
* **Frontend**: HTML, CSS, Bootstrap (via templates)
* **Database**: MySQL
* **ML/DL**: TensorFlow, Scikit-learn
* **Model Saving**: Keras `.h5`, Joblib `.pkl`

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection-flask.git
cd fraud-detection-flask
```

---

### 2. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, manually install:

```bash
pip install flask pymysql pandas numpy scikit-learn tensorflow joblib
```

---

### 4. Setup MySQL Database

#### MySQL Configuration:

Update the following in `app.py` if needed:

```python
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "frauddetectdb"
}
```

#### SQL Tables Required:

Run the frauddetect_transactions.sql file to set up the database:

---

### 5. Run the Application

```bash
python app.py
```

Then open in browser: `http://127.0.0.1:5000/`

---

## 👥 User Roles

### 👤 Remote User

* `/remote_user/register`
* `/remote_user/login`
* `/remote_user/dashboard`
* `/remote_user/find_transaction`

### 👨‍💼 Service Provider (Admin)

* `/service_provider/login` → username: `admin`, password: `admin`
* `/service_provider/dashboard`
* `/service_provider/view_users`
* `/service_provider/browse_datasets`
* `/service_provider/view_train_test_accuracy_bar_chart`
* `/service_provider/view_train_test_accuracy_results`
* `/service_provider/view_financial_trans_detection_type`

---

## 📊 Model Outputs

Trained models are saved as:

* `fraud_detection_dnn_model.h5`
* `svm_fraud_model.pkl`
* `logistic_regression_fraud_model.pkl`
* `gradient_boosting_fraud_model.pkl`

Model comparison results are stored in:

```json
model_results.json
```

And visualized via bar charts and results page.

---

## 📈 ML Pipeline Highlights

* Label Encoding for categorical features: `Gender`, `Bank_Name`, `Trans_Type`, `Location`
* Feature Scaling via `StandardScaler`
* Model Evaluation: Accuracy, Confusion Matrix, Classification Report
* Best model is auto-selected based on accuracy and used for prediction

---

## ✅ Sample Transaction Prediction Workflow

1. Login as remote user
2. Navigate to `Find Transaction`
3. Enter transaction details
4. System will:

   * Pick the most accurate model
   * Preprocess your input
   * Predict if the transaction is **Fraud** or **Not Fraud**
   * Store result in DB

---

## 📌 Notes

* Make sure your CSV file (`updated_synthetic_fraud_dataset.csv`) is in the root directory.
* Models are trained when `/browse_datasets` is accessed by admin.
* If you change dataset columns, update preprocessing accordingly.

---

## 🔒 Security Considerations

* Admin credentials are hardcoded (`admin`/`admin`)
* Passwords stored as plain text (should be hashed in production)
* No CSRF protection
* No input sanitization for SQL

**Use only for academic/demo purposes.**

---
