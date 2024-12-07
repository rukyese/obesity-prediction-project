# Obesity Level Prediction Project

## Overview
This project focuses on estimating obesity levels based on eating habits and physical conditions using Python for Data Science. The dataset includes health and dietary information from individuals in Mexico, Peru, and Colombia. The project applies data cleaning, exploratory data analysis (EDA), advanced visualizations, and machine learning techniques to predict obesity levels.

## Dataset
The dataset contains 2111 records with 17 attributes. It is labeled with the target variable `NObeyesdad`, categorizing individuals into seven obesity levels:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

### Key Features
1. `Gender` (Categorical)
2. `Age` (Continuous)
3. `Height` (Continuous)
4. `Weight` (Continuous)
5. `family_history_with_overweight` (Binary)
6. `FAVC` (Binary): High-calorie food consumption
7. `FCVC` (Continuous): Frequency of vegetable consumption
8. `NCP` (Continuous): Number of main meals daily
9. `CAEC` (Categorical): Food consumption between meals
10. `SMOKE` (Binary): Smoking habit
11. `CH2O` (Continuous): Daily water consumption
12. `SCC` (Binary): Calorie monitoring
13. `FAF` (Continuous): Physical activity frequency
14. `TUE` (Continuous): Time using technological devices
15. `CALC` (Categorical): Alcohol consumption frequency
16. `MTRANS` (Categorical): Mode of transportation
17. `NObeyesdad` (Target): Obesity level

## Project Structure
### Week 1: Data Importing and Cleaning
- Import and inspect the dataset.
- Handle missing values and duplicates.
- Encode binary and multi-class categorical variables.
- Handle outliers using the IQR method.
- Normalize continuous variables using MinMaxScaler.

### Week 2: Exploratory Data Analysis (EDA)
- Generate summary statistics for continuous variables.
- Visualize distributions with histograms and KDE plots.
- Explore relationships using boxplots.
- Analyze correlations with heatmaps.

### Week 3: Machine Learning
- Feature engineering and scaling using StandardScaler.
- Split data into training and testing sets (80:20 ratio).
- Implement Logistic Regression and Random Forest models.
- Evaluate models using metrics such as accuracy, precision, recall, and F1-score.
- Visualize feature importances and confusion matrices.

## Deliverables
1. **Code**: Clean and well-commented scripts for data preprocessing, EDA, and machine learning models.
2. **Visualizations**: Histograms, KDE plots, boxplots, heatmaps, and feature importance plots.
3. **Report**: Comprehensive documentation detailing the dataset, preprocessing steps, EDA insights, model performance, and conclusions.

## How to Run
1. Clone the repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Place the dataset (`ObesityDataSet.csv`) in the root folder.
4. Run the main script:
   ```bash
   python Obesity_Prediction_Analysis.py
   ```

## Results
- Logistic Regression and Random Forest models were implemented and evaluated.
- Visual insights and feature importance analyses were generated to interpret model predictions.
- The Random Forest model demonstrated higher performance compared to Logistic Regression based on evaluation metrics.

## Insights
- Eating habits, physical activity, and calorie monitoring are significant factors influencing obesity levels.
- Continuous features like `Weight`, `FAF`, and `Age` showed strong correlations with obesity levels.

## Future Work
- Incorporate additional models such as Support Vector Machines (SVM) or Neural Networks.
- Deploy the best-performing model as a web application for real-time obesity prediction.
- Analyze data from additional regions to improve model generalizability.

## Authors
This project was created as part of the **HubbleMind Labs Internship** program.

