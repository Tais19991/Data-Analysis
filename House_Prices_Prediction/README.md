## Boston House Price Prediction

This project aims to predict the median value of homes in Boston suburbs using a Linear Regression model. 
The dataset used is the well-known Boston Housing dataset, and various logarithmic transformations are applied to the features to improve model accuracy. 
The objective is to explore how different transformations of the data affect the performance of the linear regression model in predicting house prices.

### Data
The dataset used for this project is the Boston Housing dataset, which contains the following features:  

CRIM: Crime rate per capita  
ZN: Proportion of residential land zoned for large lots  
INDUS: Proportion of non-retail business acres per town  
CHAS: Charles River dummy variable (1 if tract bounds river, 0 otherwise)
NOX: Nitrogen oxides concentration (parts per 10 million)  
RM: Average number of rooms per dwelling  
AGE: Proportion of owner-occupied units built before 1940  
DIS: Weighted distances to employment centers  
RAD: Index of accessibility to radial highways  
TAX: Property tax rate per $10,000  
PTRATIO: Pupil-teacher ratio  
B: Proportion of residents of African American descent  
LSTAT: Percentage of lower status population  
PRICE: Median value of owner-occupied homes in $1000s (target variable)

### Model  
Linear Regression with Logarithmic Transformations  

To enhance the model's performance, several logarithmic transformations are applied to the features.  
The transformations help to handle skewed data, stabilize variance, and improve the linearity between the features and target variable. 


### Future Work
- Model Evaluation: Further evaluate the model with cross-validation and different data splits.
- Feature Engineering: Experiment with additional feature transformations or interactions.
- Advanced Models: Explore more complex regression models (e.g., Ridge or Lasso regression) to compare their performance with the current linear model.


### License
This project is licensed under the MIT License - see the LICENSE file for details.

