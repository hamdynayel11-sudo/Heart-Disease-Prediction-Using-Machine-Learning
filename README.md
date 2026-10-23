# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project explores the application of machine learning classification algorithms to predict the presence of heart disease using clinical and medical features.

The project covers Exploratory Data Analysis (EDA), feature engineering, data preprocessing, model training, hyperparameter tuning, and performance evaluation. Four classification models were compared to assess their predictive performance and generalization on unseen data.

The primary objective is to investigate how machine learning can support heart disease risk prediction in an educational setting.

> **Disclaimer:** This project is for educational purposes only. Its predictions are not intended for clinical diagnosis or medical decision-making.

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

The dataset was explored to understand its structure, feature distributions, and relationships with the target variable.

The analysis included:

* Inspecting dataset information and descriptive statistics.
* Checking for missing values and duplicate records.
* Examining the distribution of heart disease cases.
* Exploring age distribution across target classes.
* Analyzing heart disease distribution by chest pain type, exercise-induced angina, and ST slope.
* Investigating cholesterol, blood pressure, and maximum heart rate using visualizations.
* Examining potential cholesterol outliers using the Interquartile Range (IQR) method.

Visualizations were created using Matplotlib, Seaborn, and Plotly.

---

## ⚙️ 2. Feature Engineering

Three additional features were created to explore potentially useful relationships among the clinical variables.

### Estimated Maximum Heart Rate

Calculated using:

`Estimated_Max_HR = 220 - Age`

### Heart Rate Achievement Percentage

Calculated as:

`Heart_Rate_Achievement_Percentage = (MaxHR / Estimated_Max_HR) * 100`

This feature represents the percentage of the estimated maximum heart rate achieved by the patient.

### Age–Blood Pressure Interaction

Calculated using:

`Age_BP_Interaction = Age * RestingBP`

This feature represents a potential interaction between age and resting blood pressure.

---

## 🧹 3. Data Preprocessing

### Encoding Categorical Features

* `Sex` and `ExerciseAngina` were transformed using Label Encoding.
* `ChestPainType`, `RestingECG`, and `ST_Slope` were converted into dummy variables using One-Hot Encoding.

### Feature Scaling

`StandardScaler` was applied to numerical features, including:

* `Age`
* `RestingBP`
* `Cholesterol`
* `MaxHR`
* `Estimated_Max_HR`
* `Heart_Rate_Achievement_Percentage`
* `Age_BP_Interaction`

Feature scaling is particularly important for distance-based algorithms such as K-Nearest Neighbors (KNN).

### Train-Test Split

The dataset was divided into training and testing sets using the following configuration:

* **Training set:** 80%
* **Testing set:** 20%
* **Random state:** 42
* **Stratification:** Applied to preserve the target class distribution.

---

## 🤖 4. Machine Learning Models

Four classification approaches were implemented and evaluated.

### 1. Standard Logistic Regression

Used as a baseline classification model to establish a reference for comparison with the other approaches.

### 2. Polynomial Logistic Regression

Polynomial features of degree 2 were generated before training Logistic Regression to capture potential nonlinear relationships and feature interactions.

### 3. K-Nearest Neighbors (KNN)

KNN was trained using scaled features.

`GridSearchCV` with 5-fold cross-validation was used to search for the optimal number of neighbors, with candidate values ranging from `k = 1` to `k = 10`.

The selected configuration was `k = 5`.

### 4. Decision Tree Classifier

A Decision Tree classifier was used to explore nonlinear decision rules and feature-based splits.

The tree's depth was restricted to control model complexity and help reduce overfitting.

---

## 📊 5. Model Evaluation

The models were evaluated using the following metrics:

| Metric                    | Purpose                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| Accuracy                  | Measures the proportion of correctly classified observations.                                             |
| Mean Absolute Error (MAE) | Measures the average absolute difference between binary predictions and actual labels.                    |
| Confusion Matrix          | Summarizes true positives, true negatives, false positives, and false negatives.                          |
| Precision                 | Measures the proportion of positive predictions that are correct.                                         |
| Recall                    | Measures the proportion of actual positive cases correctly identified.                                    |
| F1-score                  | Combines precision and recall into a single metric.                                                       |
| ROC Curve                 | Visualizes the trade-off between true positive and false positive rates across classification thresholds. |
| ROC-AUC                   | Measures how well the model distinguishes between the two target classes.                                 |

Training and testing metrics were also compared to assess generalization and identify potential overfitting or underfitting.

---

## 🏆 6. Model Evaluation & Comparison Results

The following table summarizes the reported performance of the four models.

| Model                          | Training Accuracy | Testing Accuracy | Training MAE | Testing MAE |  ROC-AUC |
| ------------------------------ | ----------------: | ---------------: | -----------: | ----------: | -------: |
| Standard Logistic Regression   |            86.10% |           88.59% |         0.14 |        0.11 | **0.93** |
| Polynomial Logistic Regression |            90.87% |           82.61% |         0.09 |        0.17 |     0.87 |
| KNN (`k = 5`)                  |            88.15% |       **89.67%** |         0.12 |    **0.10** |     0.92 |
| Decision Tree                  |            85.56% |           80.98% |         0.14 |        0.19 |     0.88 |

*Note: These values reflect the reported evaluation results. Ensure that all metrics are calculated from the same train-test split and final model configurations.*

### Key Findings

**🥇 K-Nearest Neighbors (KNN)**

* Achieved the highest reported testing accuracy of **89.67%**.
* Recorded the lowest reported testing MAE of **0.10**.
* Achieved a ROC-AUC score of **0.92**.
* Provided a strong balance between predictive performance and generalization in this experiment.

**🥈 Standard Logistic Regression**

* Achieved a testing accuracy of **88.59%**.
* Recorded the highest ROC-AUC score among the four models at **0.93**.
* Demonstrated strong performance without the larger training-testing gap observed in Polynomial Logistic Regression.

**Polynomial Logistic Regression**

* Achieved a training accuracy of 90.87% but a lower testing accuracy of 82.61%.
* The performance gap suggests potential overfitting: the model performed better on training data than on unseen testing data.
* Adding polynomial features did not improve generalization in this experiment.

**Decision Tree**

* Achieved a testing accuracy of 80.98%.
* Recorded a testing MAE of 0.19 and a ROC-AUC score of 0.88.
* Had the lowest testing accuracy among the evaluated models.

### 🏁 Final Verdict

Based on the reported testing metrics, **KNN (`k = 5`) achieved the highest testing accuracy and lowest testing MAE**, making it the best-performing model by those two metrics in this experiment.

Standard Logistic Regression remains a strong alternative, achieving the highest ROC-AUC score of **0.93**.

The preferred model depends on the evaluation metric and the relative importance of different classification errors. In medical prediction tasks, accuracy alone is not sufficient; recall, precision, confusion matrices, and independent validation are also important.

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── heart.csv
├── Heart Disease Prediction Using Machine Learning.ipynb
├── Heart Disease Prediction Using Machine Learning.py
└── README.md
```

*The filenames above should match the actual files in the repository.*

---

## 🚀 How to Run the Project

### 1. Clone or Download the Repository

Download the repository from GitHub or clone it using Git.

### 2. Install the Required Libraries

Make sure Python is installed, then run:

```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn
```

### 3. Prepare the Dataset

Ensure that `heart.csv` is located in the project directory.

### 4. Run the Project

Open the Jupyter Notebook in Jupyter Notebook or Visual Studio Code and execute the cells sequentially.

Alternatively, run the Python script:

```bash
python "Heart Disease Prediction Using Machine Learning.py"
```

---

## ⚠️ Disclaimer

This project is intended for educational purposes and machine learning practice only.

The reported results do not establish clinical reliability. The models should not be used to diagnose heart disease, determine treatment, or replace professional medical evaluation.

Any real-world medical application would require additional validation on independent and representative clinical data, careful assessment of false negatives and false positives, and appropriate clinical oversight.
