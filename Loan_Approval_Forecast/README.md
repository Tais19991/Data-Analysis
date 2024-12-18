## Loan Approval Prediction Using Logistic Regression

This project aims to predict loan approval using a Logistic Regression model.  
The dataset consists of features related to the applicant's financial and demographic background, which are used to classify loan applications 
as "approved" or "not approved."

### Background
Loan approval decisions involve assessing an applicant's creditworthiness using various
factors such as credit score, debt ratios, and assets. 
Logistic Regression, a reliable binary classification algorithm, is used here to predict the likelihood of loan approval.

### Dataset
taken from https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

### Methodology
- Data Preprocessing
 - Handled missing values through imputation.
- Standardized numerical features for consistent scaling.
- Encoded categorical variables into numeric formats.
- Split data into training and testing sets.
- Feature Selection
- Analyzed correlations and multicollinearity to select significant features.
- Dropped features with high VIF values to avoid overfitting.
- Model Building
- Converted coefficients to odds ratios for clearer interpretation.
- Overall Conclusion

### The final model showed:

- High Accuracy: The model has a high accuracy for both the training (92.0%) and test sets (89.1%),
suggesting it can distinguish between approved and not-approved loans effectively.  

- Low Misclassification: A relatively low misclassification rate (8.0%) on the test set indicates that the model's predictions are reliable.  

The logistic regression model for loan approval performs well, showing strong predictive capability on both training and test datasets 
with minimal overfitting. While the false positive and false negative counts are relatively low, depending on the business context 
(e.g., the cost of a wrongly approved loan), additional steps like adjusting decision thresholds or adding more informative features could be considered to reduce errors further

<img src="..Animated_Charts/gifs/logistic_regression_loan.gif" alt="Alternate text" width="700"/>

#### By examining this equation, we gain valuable insights into the factors affecting loan approval:

**CIBIL Score: The Primary Factor**  
With a coefficient of 4.28, the CIBIL Score is the most influential factor in the model. Every increase in your score significantly boosts your chances of approval, making it the most critical variable in the decision-making process.

**Debt-to-Income Ratio: A Complex Relationship**  
A coefficient of 0.84 suggests that higher Debt-to-Income Ratios may sometimes lead to approvals. While this is not what we typically expect—lenders usually prefer lower ratios—it may reflect certain trends in the data, such as approvals for high-income applicants who can manage larger debts.

**Income per Person: A Supporting Factor**  
This factor has a smaller positive effect on loan approval. While it plays a role, its influence is relatively minor compared to other factors like the CIBIL Score.

**All Assets, Debt-to-Assets Ratio, and Loan Term: The Barriers**  
These factors negatively impact the odds of approval. Among them, Loan Term has the strongest negative effect, with longer terms reducing the likelihood of approval.


### Future Work
- Model Enhancement: Consider advanced models like Random Forest or Gradient Boosting for better accuracy.
- Feature Engineering: Explore interaction terms and new features.
- Validation: Use cross-validation for a more reliable model assessment.
