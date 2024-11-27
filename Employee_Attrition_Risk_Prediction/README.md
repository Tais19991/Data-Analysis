## Employee Attrition Risk Prediction (Multiple Models)

### Overview
Employee attrition is a critical challenge for businesses, as the recruitment and onboarding process is both costly and time-intensive. This project predicts whether an employee is likely to leave (attrition) using machine learning models. By identifying the key factors influencing attrition, businesses can improve retention strategies and reduce recruitment costs.

### Business Problem
1.**Problem Statement:**
Hiring a new employee can cost up to 20% of the employee's annual salary. This project addresses the challenge of predicting attrition by analyzing employee data and building models to proactively identify high-risk employees.

2. **Dataset:**
The project uses the "Human Resources" dataset, which includes various employee attributes, such as age, monthly income, job role, and work-life balance ratings.

### Workflow

- Data Exploration and Cleaning
- Analyzed key trends in attrition vs. retained employees.
- Removed irrelevant or constant columns (EmployeeCount, StandardHours, etc.).
- Performed EDA with visualizations like histograms, KDE plots, and heatmaps.
- Converted categorical variables (e.g., OverTime, Attrition) into numerical representations.
  
#### Feature Selection

- Examined correlations between features and the target variable.
- Identified multicollinearity using Variance Inflation Factor (VIF).
- Selected relevant predictors based on statistical significance (p-values).

#### Modeling

- Logistic Regression
- Random Forest Classifier
- Deep Learning (Neural Network)
- Evaluated model performance using accuracy, precision, recall, and F1-score.
  
#### Class Imbalance Handling

- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- Conducted undersampling of the majority class to further address imbalance.
  
### Key Findings
**Insights from Data Analysis:**
- Employees with lower monthly incomes and fewer years at the company are more likely to leave.
- Proximity to workplace and job satisfaction are positively correlated with retention.
- Certain roles (e.g., Sales Representatives, Lab Technicians) show higher attrition risks.
  
**Performance:**
- Logistic Regression achieved 87.4% accuracy, with strong precision for retained employees.
- Random Forest yielded more robust feature importance analysis but struggled with low recall for attrition cases.
- Deep Learning improved predictive performance but was still limited by the class imbalance in the dataset.
- Impact of Balancing: Data balancing significantly improved recall for attrition (from 8% to ~50% in some cases) but slightly reduced accuracy for retained employees.

### Tools and Technologies
- Programming Language: Python
- Libraries: pandas, scikit-learn, TensorFlow, seaborn, matplotlib, imbalanced-learn, statsmodels
- Data Handling: SMOTE, VIF analysis, dummy variable encoding
- Visualization: Correlation heatmaps, boxplots, histograms, bar plots
  
### Future Enhancements
- Incorporate additional data points, such as performance appraisals or team dynamics, to improve model precision.
- Experiment with ensemble methods like Gradient Boosting or XGBoost for improved generalization.
- Develop interpretability tools (e.g., SHAP or LIME) to provide actionable insights for HR teams.
  
### Usage

1. Clone the repository and install dependencies:
`git clone https://github.com/Tais19991/Data-Analysis/edit/main/Employee_Attrition_Risk_Prediction`
`cd Employee_Attrition_Risk_Prediction`

2. Run the notebook:
`jupyter notebook Employee_Attrition_Risk_Prediction(Multiple_Models).ipynb`

3. Customize parameters (e.g., sampling strategy, number of epochs) to explore the model behavior.
