# ARTI-308 Machine Learning Lab-2


## 1. Lab Overview
The goal of this Lab is to identify a real-world machine learning problem and implement a structured methodology to solve it. I have selected a **Binary Classification** problem to predict the onset of diabetes based on diagnostic measurements.

## 2. Dataset Summary
- **Source**: Pima Indians Diabetes Database (via Kaggle).
- **Type**: Tabular Data.
- **Features**: Includes medical metrics such as Glucose levels, BMI, Blood Pressure, and Age.
- **Target**: `Outcome` (1 for diabetes, 0 for no diabetes).

## 3. Methodology Diagram
The project follows a linear machine learning workflow to ensure logical consistency and data quality:



## 4. Implementation Steps
- **Data Acquisition**: The dataset is fetched automatically using the `kagglehub` library to ensure the latest version is used.
- **Preprocessing**: Inconsistent zero values in critical features (Glucose, BMI, Blood Pressure) are replaced with the mean value to maintain data integrity.
- **Train/Test Split**: The data is divided into 80% for training and 20% for testing.
- **Modeling**: A Logistic Regression model was chosen for its efficiency in binary classification tasks.
- **Evaluation**: The model performance is measured using the Accuracy Score.

## 5. Final Results
After executing the methodology in a Python environment, the model achieved the following performance:
- **Final Accuracy: 76.62%**

## 6. Conclusion
The methodology successfully demonstrates how defining a clear workflow—from data cleaning to model evaluation—leads to a reliable predictive model. The logic and structured approach ensure that the problem is addressed systematically as per the lab requirements.
