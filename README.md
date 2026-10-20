# Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project explores the application of machine learning classification algorithms to predict the presence of heart disease using clinical and medical features.

The project includes Exploratory Data Analysis (EDA), feature engineering, data preprocessing, model training, and evaluation. Four classification approaches are compared to identify models that achieve strong predictive performance on the test dataset.

**The goal is to investigate how machine learning can support heart disease prediction. This project is educational and is not intended for clinical diagnosis.**

---

## 📂 Dataset

The project uses `heart.csv`, which contains patient-related clinical features and a binary target variable.

### Target Variable

| Feature        | Description                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| `HeartDisease` | Target variable: `1` indicates heart disease, while `0` indicates no heart disease. |

### Main Features

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `Age`            | Patient's age                         |
| `Sex`            | Patient's sex                         |
| `ChestPainType`  | Type of chest pain                    |
| `RestingBP`      | Resting blood pressure                |
| `Cholesterol`    | Serum cholesterol level               |
| `FastingBS`      | Fasting blood sugar indicator         |
| `RestingECG`     | Resting electrocardiogram results     |
| `MaxHR`          | Maximum heart rate achieved           |
| `ExerciseAngina` | Exercise-induced angina indicator     |
| `Oldpeak`        | ST depression induced by exercise     |
| `ST_Slope`       | Slope of the peak exercise ST segment |

---

## 🛠️ Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn

---

## 🔍 1. Exploratory Data Analysis (EDA)

The dataset was explored to understand its structure, distributions, and relationships between clinical features and the target variable.

The analysis included:

* Inspecting dataset information and descriptive statistics.
* Checking for missing values and duplicate records.
* Examining the distribution of heart disease cases.
* Exploring age distribution across target classes.
* Analyzing heart disease distribution by chest pain type, exercise-induced angina, and ST slope.
* Investigating cholesterol, blood pressure, and maximum heart rate using visualizations.
* Examining potential cholesterol outliers using the Interquartile Range (IQR) method.

Visualizations were created using Seaborn, Matplotlib, and Plotly.

---

## ⚙️ 2. Feature Engineering

Three additional features were created to explore potentially useful relationships in the data.

### Estimated Maximum Heart Rate

Calculated using:

`Estimated_Max_HR = 220 - Age`

### Heart Rate Achievement Percentage

Calculated as the patient's maximum heart rate divided by the estimated maximum heart rate, multiplied by 100.

`Heart_Rate_Achievement_Percentage = (MaxHR / Estimated_Max_HR) * 100`

### Age–Blood Pressure Interaction

`Age_BP_Interaction = Age * RestingBP`

These features were introduced to represent additional relationships between age, blood pressure, and heart rate.

---

## 🧹 3. Data Preprocessing

The following preprocessing steps were applied:

### Encoding Categorical Features

* `Sex` and `ExerciseAngina` were transformed using Label Encoding.
* `ChestPainType`, `RestingECG`, and `ST_Slope` were converted into dummy variables using One-Hot Encoding.

### Feature Scaling

StandardScaler was applied to the following numerical features:

* Age
* RestingBP
* Cholesterol
* MaxHR
* Estimated_Max_HR
* Heart_Rate_Achievement_Percentage
* Age_BP_Interaction

### Train-Test Split

The dataset was split into training and testing sets using:

* **Training set:** 80%
* **Testing set:** 20%
* **Random state:** 42
* **Stratification:** Applied to preserve the target class distribution.

---

## 🤖 4. Machine Learning Models

Four classification approaches were implemented and evaluated.

### 1. Logistic Regression

A standard Logistic Regression classifier was used as the baseline model.

### 2. Logistic Regression with Polynomial Features

Polynomial features of degree 2 were generated to introduce nonlinear feature combinations before training Logistic Regression.

This approach was investigated to determine whether additional feature interactions could improve predictive performance.

### 3. K-Nearest Neighbors (KNN)

KNN was trained using scaled features.

GridSearchCV with 5-fold cross-validation was used to search for the best number of neighbors from `k = 1` to `k = 10`.

The final KNN model was trained using the selected value of `k`.

### 4. Decision Tree Classifier

A Decision Tree classifier was trained with:

* `random_state = 42`
* `max_depth = 4`

The depth restriction was used to limit tree complexity.

---

## 📊 5. Model Evaluation

The models were evaluated using several classification metrics:

| Metric           | Purpose                                                                                |
| ---------------- | -------------------------------------------------------------------------------------- |
| Accuracy         | Measures the proportion of correctly classified observations.                          |
| MAE              | Measures the average absolute difference between binary predictions and actual labels. |
| Confusion Matrix | Shows true positives, true negatives, false positives, and false negatives.            |
| Precision        | Measures the proportion of positive predictions that are correct.                      |
| Recall           | Measures the proportion of actual positive cases identified by the model.              |
| F1-score         | Combines precision and recall into a single metric.                                    |
| ROC Curve        | Visualizes the trade-off between true positive and false positive rates.               |
| ROC-AUC          | Measures how well the model distinguishes between the two target classes.              |

Training and testing accuracy were also compared to investigate potential overfitting.

---

## 🏆 6. Results & Model Comparison

The following results are reported in the project's final evaluation section.

| Model                                        | Test Accuracy | ROC-AUC |
| -------------------------------------------- | ------------: | ------: |
| Logistic Regression                          |        88.59% |    0.93 |
| KNN (K = 5)                                  |        88.59% |    0.93 |
| Logistic Regression with Polynomial Features |        85.87% |    0.91 |
| Decision Tree                                |        84.24% |    0.87 |

### Key Findings

* **Logistic Regression** achieved 88.59% test accuracy and an ROC-AUC of 0.93.
* **KNN (K = 5)** achieved the same reported test accuracy and ROC-AUC as standard Logistic Regression.
* Polynomial Logistic Regression did not outperform the standard Logistic Regression model in the reported results.
* The Decision Tree achieved lower test accuracy and ROC-AUC than the two leading models.

Based on the reported results, standard Logistic Regression and KNN were the top-performing models in this comparison.

**Important:** These results represent the project's reported evaluation and do not establish that either model is clinically reliable. Further validation, particularly with independent data, would be necessary before considering any real-world medical application.

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── heart.csv
├── heart_disease_prediction.ipynb
└── README.md
```

*Update the notebook filename above to match the actual filename in your repository.*

---

## 🚀 How to Run the Project

1. Clone or download this repository.

2. Make sure Python is installed.

3. Install the required libraries:

   ```bash
   pip install pandas numpy seaborn matplotlib plotly scikit-learn
   ```

4. Ensure `heart.csv` is located in the project directory.

5. Open the Jupyter Notebook.

6. Run the cells sequentially to perform EDA, preprocessing, model training, and evaluation.

---

## ⚠️ Disclaimer

This project is for educational and machine learning practice purposes only. Its predictions should not be used to diagnose heart disease, make treatment decisions, or replace professional medical evaluation.
