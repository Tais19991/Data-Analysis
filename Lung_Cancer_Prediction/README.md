## Lung Cancer Risk Factors Analysis

### Introduction 
This study, published in Nature Medicine, analyzed data from over 1,000 individuals in China over an average of six years.  
Participants were divided into two groups based on their air pollution exposure: high and low.
Dataset Source: Kaggle

### Dataset Overview
The dataset includes information on lung cancer patients, encompassing variables such as:
Age, Gender, Air Pollution Exposure, Alcohol Use, Dust Allergy, Occupational Hazards, Genetic Risk
Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing of Blood
Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Fingernails
Snoring

### Data Exploration
The dataset contains 1,000 entries with no duplicate values.  
Key variables were analyzed to assess correlations and distributions, particularly focusing on lung cancer risk factors and their associations.


### Visual Data Exploration
- Age Distribution: The analysis included visualizations of age distribution among patients, highlighting differences by gender.
- Genetic and Environmental Risk Factors: Relationships between genetic risk and environmental exposures were explored.
- Lifestyle and Chronic Disease Factors: The distribution of various lifestyle factors was examined in relation to lung cancer risk levels.
- Symptoms and Lung Cancer Risk Level: Severity of symptoms was assessed against cancer risk levels.

### Linear Regression Analysis
A linear regression model was developed to predict the risk of lung cancer based on the identified risk factors.   
The model achieved an R-squared value of 0.92, indicating a strong explanatory power. 

### Ordinal Prediction Model
In addition to the linear regression model, an ordinal prediction model was developed to categorize patients based on the severity of their 
lung cancer risk. This model achieved an accuracy of 1.0, demonstrating perfect classification of the training dataset.  
