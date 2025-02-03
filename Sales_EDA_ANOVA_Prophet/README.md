## Sales Data Analysis and Prediction
This project focuses on analyzing and forecasting sales data using a combination of exploratory data analysis (EDA), ANOVA testing, 
and the Prophet time-series forecasting model. 

**It consists of two key notebooks:**

- Sales_EDA_Anova.ipynb: A comprehensive exploration of the dataset, including descriptive statistics, data visualizations, and ANOVA testing to analyze key sales drivers.
- Sales_Prediction_Prophet_Model.ipynb: A robust implementation of the Prophet model for time-series forecasting, enabling accurate sales predictions and insights into trends and seasonality.

### Project Objectives
Perform detailed exploratory analysis of sales data to uncover patterns and trends.
Use ANOVA to test hypotheses about the factors affecting sales.
Build a time-series forecasting model using Prophet to predict future sales and identify critical trends.

### Notebooks Overview
1. Sales_EDA_Anova.ipynb
Purpose: Perform exploratory data analysis and test hypotheses about sales-related factors.
Key Components:
- Descriptive statistics and visualizations of sales data.
- Hypothesis testing using ANOVA to compare group means across categories.
- Insights into sales distribution, customer behavior, and product performance.

2. Sales_Prediction_Prophet_Model.ipynb
Purpose: Use Facebook's Prophet model to forecast sales and analyze trends.
Key Components:
- Time-series decomposition to identify overall trends, seasonal effects, and residuals.
- Model training and evaluation using sales data.
- Forecast visualization and diagnostics for model accuracy.

### Tools and Libraries Used

**Python Libraries:**  
- pandas, numpy: Data manipulation and analysis.  
- matplotlib, seaborn: Data visualization.  
- scipy.stats: Statistical testing (ANOVA).  
-  fbprophet (now prophet): Time-series forecasting.  
 
**Jupyter Notebooks:**  
For step-by-step interactive data analysis.  

### How to Use
Clone this repository:
`git clone https://github.com/yourusername/Sales_EDA_ANOVA_Prophet.git`

Install required dependencies:  
`pip install -r requirements.txt`  
Open the notebooks in Jupyter:  
`jupyter notebook`  
Follow the analysis steps in each notebook.  

### Results
**Exploratory Data Analysis (EDA) Results**  
The dataset contains 1,017,209 entries spanning 3 years (2013–2015), detailing sales and customer performance across 1,115 stores.  

**Key observations include:**  
- Sales Patterns: Average daily sales were €6,955, with a range from €0 to €41,551. Variability was influenced by factors like promotions, holidays, and seasonality.  
- Customer Metrics: The daily average customer count was 762, ranging up to 7,388, with trends closely tied to sales performance.   
- Promotions: Promotions were active in 44.6% of the records, significantly boosting sales during these periods.  
- Holiday Influence: Sales varied during special holidays (e.g., Christmas) and school holidays, highlighting the impact of external events on purchasing behavior.  

**Variance Analysis Results**  
Variance analysis focused on identifying differences across store types and assortment categories:  
- Levene’s Test Results: Significant differences in variances for both store types and assortments (p < 0.05), justifying the use of Welch’s ANOVA.
Welch’s ANOVA Findings:  
- Assortments: Sales patterns varied significantly across assortment types (F = 704.91, p < 0.001).
- Store Types: Significant sales differences were also identified between store types (F = 2,257.91, p < 0.001).

**Time-Series Forecasting with Prophet**  
A Prophet-based time-series model was developed to predict daily sales for each of the 1,115 stores while capturing their 
unique seasonal dynamics and slight parameter variations. 
Key results include:
- Mean Absolute Percentage Error (MAPE): ~12.7%
- Mean Absolute Error (MAE): ~865
- Root Mean Square Error (RMSE): ~1,079
- Optimized Parameters: The model utilized linear growth, multiplicative seasonality, and weekly seasonality. A changepoint prior scale of ~0.039 was found to be optimal.  
- Forecast Accuracy: The model effectively captured seasonal and weekly trends, making reliable sales forecasts for each store. However, extreme fluctuations (e.g., during holidays) presented slight challenges.  

### Future Work  
- Enhance the dataset with additional features (e.g., economic indicators).    
- Explore advanced models for forecasting (e.g., ARIMA, LSTM).   

### License
This project is licensed under the MIT License. See the LICENSE file for details.  
