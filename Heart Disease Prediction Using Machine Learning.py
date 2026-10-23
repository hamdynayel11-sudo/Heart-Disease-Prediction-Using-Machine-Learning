# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

# %%
df = pd.read_csv('heart.csv')

# %%
df.head(10)

# %% [markdown]
# # 🩺 Heart Disease Dataset Reference Guide
# 
# ---
# 
# ### 📊 1. Vital Signs & Clinical Thresholds
# 
# | Feature | Full Name & Description | Normal Range (Safe) | High Risk Threshold |
# | :--- | :--- | :--- | :--- |
# | **`RestingBP`** | Resting Systolic Blood Pressure ($mm\,Hg$) | `< 120` | `≥ 140` (Hypertension) |
# | **`Cholesterol`** | Serum Cholesterol ($mg/dl$) | `< 200` | `≥ 240` (Hypercholesterolemia) |
# | **`FastingBS`** | Fasting Blood Sugar ($>120\,mg/dl$) | `0` (Normal, `< 120`) | `1` (High, `> 120`) |
# | **`MaxHR`** | Maximum Heart Rate Achieved | `220 - Age` (Expected) | Exceptionally low rate during exercise |
# 
# ---
# 
# ### 🫀 2. Symptoms & Diagnostic Features
# 
# * **`ChestPainType`** (Chest Pain Classification)
#   * **`TA` (Typical Angina):** **Highest Risk** — Classic exertional chest pressure/pain.
#   * **`ATA` (Atypical Angina):** Moderate Risk — Non-classic cardiac discomfort.
#   * **`NAP` (Non-Anginal Pain):** Low Risk — Non-cardiac chest pain.
#   * **`ASY` (Asymptomatic):** Silent Ischemia — No pain, but heart disease can still be present (common in diabetics).
# 
# * **`ExerciseAngina`** (Exercise-Induced Angina)
#   * **`Y` (Yes):** High Risk — Indicates restricted coronary blood flow during activity.
#   * **`N` (No):** Normal.
# 
# ---
# 
# ### 📉 3. Electrocardiogram (ECG) & Stress Test Indicators
# 
# * **`RestingECG`** (Resting Electrocardiogram)
#   * **`Normal`:** Healthy electrical conduction.
#   * **`ST` / `LVH`:** **High Risk** — Indicates ST-T wave abnormalities or Left Ventricular Hypertrophy (heart muscle strain).
# 
# * **`Oldpeak`** (ST Depression induced by exercise)
#   * **`0` (or near 0):** Normal.
#   * **Higher positive values:** **High Risk** — Significant ST depression signals myocardial ischemia (coronary blockage).
# 
# * **`ST_Slope`** (Slope of Peak Exercise ST Segment)
#   * **`Up` (Upsloping):** Normal / Healthy cardiac response.
#   * **`Flat` / `Down` (Flat/Downsloping):** **High Risk** — Strong clinical indicator of Coronary Artery Disease (CAD).
# 
# --- 
# 
# ### 📉 4. Traget Classes 
# 
# * **`HeartDisease`** (Heart Disease)
#   * **`1`:** heart disease.
#   * **`0`:** Normal.

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ## EDA

# %%
print(df.duplicated().sum())

# %%
print(df.isnull().sum())

# %%
df['HeartDisease'].value_counts()

# %%
df['HeartDisease'].value_counts(normalize = True).round(2) 

# %%
sns.histplot(df, x = 'Age', kde = True, hue = 'HeartDisease')
plt.title("Age Distribution")
plt.legend(title='Heart Disease', labels=['Normal (0)', 'Heart Disease (1)'])
plt.show()

# %%
sns.countplot(data = df, x = 'ChestPainType', hue = 'HeartDisease')
plt.title('Heart Disease Distribution by Chest Pain Type')
plt.show()

# %%
sns.countplot(data = df, x = 'ExerciseAngina', hue = 'HeartDisease')
plt.title("Impact of Exercise-Induced Angina on Heart Disease Risk")
plt.show()

# %%
sns.countplot(data = df, x = 'ST_Slope', hue = 'HeartDisease')
plt.title("Heart Disease Risk Across ST Segment Slopes")
plt.show()

# %%
fig = px.box(
    df,
    x="HeartDisease",
    y="Cholesterol",
    color="HeartDisease",
    title='Cholesterol Level Distribution by Heart Disease Status')
fig.show()

# %%
Q1 = df['Cholesterol'].quantile(0.25)
Q3 = df['Cholesterol'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bounde = Q3 + 1.5 * IQR

cholesterol = df[(df['Cholesterol'] >= lower_bound) & (df['Cholesterol'] <= upper_bounde)]

# %%
fig = px.box(
    cholesterol,
    x="HeartDisease",
    y="Cholesterol",
    color="HeartDisease",
    title='Cholesterol Level Distribution by Heart Disease Status')
fig.show()

# %%
sns.scatterplot(data = df, x = 'Age', y = 'MaxHR', hue = 'HeartDisease')
plt.legend(title='Heart Disease')
plt.title("Max Heart Rate vs Age Categorized by Heart Disease")
plt.show()

# %%
sns.scatterplot(data = df, x = 'Cholesterol', y = 'RestingBP', hue = 'HeartDisease')
plt.legend(title='Heart Disease')
plt.title("Resting Blood Pressure vs Cholesterol Levels by Heart Disease")
plt.show()

# %%
sns.scatterplot(data = df, x = 'Age', y = 'RestingBP', hue = 'HeartDisease')
plt.legend(title='Heart Disease')
plt.title("Age vs Resting Blood Pressure by Heart Disease")
plt.show()

# %% [markdown]
# ## Feature Engineering

# %%
df['Estimated_Max_HR'] = 220 - df['Age']
df['Heart_Rate_Achievement_Percentage'] = (df['MaxHR'] / df['Estimated_Max_HR']).round(2) * 100
df['Age_BP_Interaction'] = df['Age'] * df['RestingBP']

# %% [markdown]
# ## Data Preprocessing

# %%
label_encoding = LabelEncoder()

df['gender_encoded'] = label_encoding.fit_transform(df['Sex'])
df['Exercise_Angina_encoded'] = label_encoding.fit_transform(df['ExerciseAngina'])

# %%
encoding = df.copy()

columns = ['ChestPainType', 'RestingECG', 'ST_Slope']

encoding = pd.get_dummies(encoding, columns=columns, dtype=int)

# %% [markdown]
# ## Splitting Data  

# %%
drop_columns = ['HeartDisease', 'Sex', 'ExerciseAngina']

X = encoding.drop(columns=drop_columns, errors='ignore')

y = encoding['HeartDisease']

# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

# %%
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

# %% [markdown]
# ## Models Construction 

# %% [markdown]
# ### 1. Logistic model

# %%
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(x_train_scaled, y_train)

# %%
y_prediction_logistic = logistic_model.predict(x_test_scaled)

# %%
# Predictions on training data
y_train_logistic = logistic_model.predict(x_train_scaled)

# Predictions on testing data
y_test_logistic = logistic_model.predict(x_test_scaled)

# Calculate errors
logisitc_train_mae = mean_absolute_error(y_train, y_train_logistic)
logistic_test_mae = mean_absolute_error(y_test, y_test_logistic)

print("Training MAE:", round(logisitc_train_mae, 2))
print("Testing MAE:", round(logistic_test_mae, 2))

logistic_trainig_accuracy = 1 - logisitc_train_mae
logistic_testing_accuracy = 1 - logistic_test_mae

print("Training Accuracy", round(logistic_trainig_accuracy, 2))
print("Testing Accuracy", round(logistic_testing_accuracy, 2))

# %%
y_train_logistic = logistic_model.predict(x_train_scaled)
y_test_logistic = logistic_model.predict(x_test_scaled)

train_accuracy_logistic = accuracy_score(y_train, y_train_logistic)
test_accuracy_logistic = accuracy_score(y_test, y_test_logistic)

print(f"Training Accuracy For Logistic Model : {train_accuracy_logistic:.2%}")
print(f"Testing Accuracy For Logistic Model: {test_accuracy_logistic:.2%}")
print(f"Difference: {(train_accuracy_logistic - test_accuracy_logistic):.2%}")

# %%
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_prediction_logistic))
print("\nClassification Report:")
print(classification_report(y_test, y_prediction_logistic))

# %%
# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_prediction_logistic)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# ROC Curve and AUC Score
y_prob_logistic = logistic_model.predict_proba(x_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_logistic)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc_score = roc_auc_score(y_test, y_prob_logistic)
print(f"AUC Score: {auc_score:.2f}")

# %% [markdown]
# ### 2. Logistic Model backed by polynomial 

# %%
logistic_poly = PolynomialFeatures(degree = 2)

x_train_poly = logistic_poly.fit_transform(x_train_scaled)
x_test_poly = logistic_poly.transform(x_test_scaled)

# %%
logistic_model_poly = LogisticRegression(max_iter=1000, random_state=42)
logistic_model_poly.fit(x_train_poly, y_train)

# %%
y_logistic_poly = logistic_model_poly.predict(x_test_poly)

# %%
# Predictions on training data
y_train_logistic_poly = logistic_model_poly.predict(x_train_poly)

# Predictions on testing data
y_test_logistic_poly = logistic_model_poly.predict(x_test_poly)

# Calculate errors
logisitc_ploy_train_mae = mean_absolute_error(y_train, y_train_logistic_poly)
logistic_poly_test_mae = mean_absolute_error(y_test, y_test_logistic_poly)

print("Training MAE:", round(logisitc_ploy_train_mae, 2))
print("Testing MAE:", round(logistic_poly_test_mae, 2))

logistic_poly_trainig_accuracy = 1 - logisitc_ploy_train_mae
logistic_poly_testing_accuracy = 1 - logistic_poly_test_mae

print("Training Accuracy", round(logistic_poly_trainig_accuracy, 2))
print("Testing Accuracy", round(logistic_poly_testing_accuracy, 2))

# %%
y_train_logistic_poly = logistic_model_poly.predict(x_train_poly)
y_test_logistic_poly = logistic_model_poly.predict(x_test_poly)

train_accuracy_logistic_poly = accuracy_score(y_train, y_train_logistic_poly)
test_accuracy_logistic_poly = accuracy_score(y_test, y_test_logistic_poly)

print(f"Training Accuracy For Logistic Model (poly) : {train_accuracy_logistic_poly:.2%}")
print(f"Testing Accuracy For Logistic Model (poly): {test_accuracy_logistic_poly:.2%}")
print(f"Difference: {(train_accuracy_logistic_poly - test_accuracy_logistic_poly):.2%}")

# %%
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_logistic_poly))
print("\nClassification Report:")
print(classification_report(y_test, y_logistic_poly))

# %%
# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_logistic_poly)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# ROC Curve and AUC Score
y_poly = logistic_model_poly.predict_proba(x_test_poly)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_poly)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc_score = roc_auc_score(y_test, y_poly)
print(f"AUC Score: {auc_score:.2f}")

# %% [markdown]
# ### 3. KNN

# %%
param_grid = {'n_neighbors': range(1, 11)}  # Try k from 1 to 10
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)  # 5-fold cross-validation
grid_search.fit(x_train_scaled, y_train)

# %%
best_k = grid_search.best_params_['n_neighbors']
print(f"Best value for k: {best_k}")

# %%
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train_scaled, y_train)

# %%
y_knn = knn_best.predict(x_test_scaled)

# %%
# Predictions on training data
y_train_knn = knn_best.predict(x_train_scaled)

# Predictions on testing data
y_test_knn = knn_best.predict(x_test_scaled)

# Calculate errors
knn_train_mae = mean_absolute_error(y_train, y_train_knn)
knn_test_mae = mean_absolute_error(y_test, y_test_knn)

print("Training MAE:", round(knn_train_mae, 2))
print("Testing MAE:", round(knn_test_mae, 2))

knn_trainig_accuracy = 1 - knn_train_mae
knn_testing_accuracy = 1 - knn_test_mae

print("Training Accuracy", round(knn_trainig_accuracy, 2))
print("Testing Accuracy", round(knn_testing_accuracy, 2))

# %%
y_train_knn = knn_best.predict(x_train_scaled)
y_test_knn = knn_best.predict(x_test_scaled)

train_accuracy_knn = accuracy_score(y_train, y_train_knn)
test_accuracy_knn = accuracy_score(y_test, y_test_knn)

print(f"Training Accuracy For KNN : {train_accuracy_knn:.2%}")
print(f"Testing Accuracy For KNN: {test_accuracy_knn:.2%}")
print(f"Difference: {(train_accuracy_knn - test_accuracy_knn):.2%}")

# %%
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_knn))
print("\nClassification Report:")
print(classification_report(y_test, y_knn))

# %%
# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_knn)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# ROC Curve and AUC Score
y_pre_knn = knn_best.predict_proba(x_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pre_knn)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc_score = roc_auc_score(y_test, y_pre_knn)
print(f"AUC Score: {auc_score:.2f}")

# %% [markdown]
# ### 4. Decision Tree

# %%
param_grid_tree = {'max_depth': range(1, 11)}  
decision_tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(decision_tree, param_grid_tree, cv=5)  
grid_search_tree.fit(x_train, y_train)

# %%
best_depth = grid_search_tree.best_params_['max_depth']
print(f"Best value for depth: {best_depth}")

# %%
tree_model = DecisionTreeClassifier(random_state=42, max_depth=best_depth)

tree_model.fit(x_train, y_train)

# %%
y_tree_prediction = tree_model.predict(x_test)

# %%
# Predictions on training data
y_train_tree = tree_model.predict(x_train)

# Predictions on testing data
y_test_tree = tree_model.predict(x_test)

# Calculate errors
tree_train_mae = mean_absolute_error(y_train, y_train_tree)
tree_test_mae = mean_absolute_error(y_test, y_test_tree)

print("Training MAE:", round(tree_train_mae, 2))
print("Testing MAE:", round(tree_test_mae, 2))

tree_trainig_accuracy = 1 - tree_train_mae
tree_testing_accuracy = 1 - tree_test_mae

print("Training Accuracy", round(tree_trainig_accuracy, 2))
print("Testing Accuracy", round(tree_testing_accuracy, 2))

# %%
y_train_tree = tree_model.predict(x_train)
y_test_tree = tree_model.predict(x_test)

train_accuracy_tree = accuracy_score(y_train, y_train_tree)
test_accuracy_tree = accuracy_score(y_test, y_test_tree)

print(f"Training Accuracy For Decision Tree : {train_accuracy_tree:.2%}")
print(f"Testing Accuracy For Decision Tree : {test_accuracy_tree:.2%}")
print(f"Difference: {(train_accuracy_tree - test_accuracy_tree):.2%}")

# %%
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_tree_prediction))
print("\nClassification Report:")
print(classification_report(y_test, y_tree_prediction))

# %%
# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_tree_prediction)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# ROC Curve and AUC Score
y_tree = tree_model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_tree)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc_score = roc_auc_score(y_test, y_tree)
print(f"AUC Score: {auc_score:.2f}")