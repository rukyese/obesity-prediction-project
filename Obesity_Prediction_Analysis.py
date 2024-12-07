# ==================== Import Libraries ====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== WEEK 1: Data Import and Cleaning ====================
# Task 1: Import the dataset and inspect its structure
df = pd.read_csv('ObesityDataSet.csv')
df2 = pd.read_csv('ObesityDataSet.csv')  # Duplicate for comparison

print("Initial Data Overview:")

# Remove duplicate records
df.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)

# Inspect the first few rows and data types
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Task 2: Data Type Conversion and Encoding
# Label Encoding binary categorical variables
label_encoder = LabelEncoder()
binary_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# One-Hot Encoding multi-class categorical variables
multi_class_columns = ['CAEC', 'CALC', 'MTRANS']
df = pd.get_dummies(df, columns=multi_class_columns)

# One-Hot Encoding the target variable 'NObeyesdad'
df = pd.get_dummies(df, columns=['NObeyesdad'])

# Inspect the transformed data
print("\nData After Encoding:")
print(df.head())
print(df.info())

# Task 3: Outlier Detection and Handling
# Handling outliers for Height and Weight using the IQR method
for col in ['Height', 'Weight']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

print("\nSummary Statistics After Outlier Handling:")
print(df[['Height', 'Weight']].describe())

# Task 4: Normalization/Standardization
# Normalizing continuous variables using MinMaxScaler
scaler = MinMaxScaler()
continuous_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

print("\nSummary Statistics After Normalization:")
print(df[continuous_columns].describe())

# ==================== WEEK 2: Exploratory Data Analysis (EDA) ====================
# Task 1: Summary Statistics
print("\nSummary Statistics for Continuous Variables:")
print(df.describe().to_string())

# Task 2: Distribution Analysis
for column in continuous_columns[:3]:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Task 3: Relationship Exploration
plt.figure(figsize=(10, 6))
sns.boxplot(x='NObeyesdad', y='Weight', data=df2, palette='coolwarm', hue='NObeyesdad', legend=False)
plt.title('Weight vs. Obesity Level', fontsize=16)
plt.xlabel('Obesity Level', fontsize=12)
plt.ylabel('Weight', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='NObeyesdad', y='FAF', data=df2, palette='viridis', hue='NObeyesdad', legend=False)
plt.title('Physical Activity Frequency (FAF) vs. Obesity Level', fontsize=16)
plt.xlabel('Obesity Level', fontsize=12)
plt.ylabel('Physical Activity Frequency (FAF)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 4: Correlation Analysis
plt.figure(figsize=(12, 8))
corr_matrix = df[continuous_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Continuous Features')
plt.show()

# ==================== WEEK 3: Advanced Visualizations and Machine Learning ====================
# Step 1: Feature Engineering and Scaling
X = df.drop(columns=[col for col in df.columns if col.startswith('NObeyesdad')])
y = label_encoder.fit_transform(df2['NObeyesdad'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log))

# Step 4: Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Step 5: Feature Importance Plot
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Feature Importances (Random Forest)')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Step 6: Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix Heatmap - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
