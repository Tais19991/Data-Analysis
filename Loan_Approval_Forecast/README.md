## Loan Approval Predictor – Streamlit App

This project predicts loan approval based on financial and demographic data provided by the user. It combines machine learning, feature engineering, and model explainability into an interactive app.

### Dataset
taken from https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

### Project Highlights:
- EDA & Model Benchmarking: Performed exploratory data analysis and compared multiple models (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM).
- Advanced Preprocessing: Pipelines include custom transformers for feature creation and data cleaning, with scaling and encoding tailored for each model.
- Explainability Tools: Integrated SHAP and LIME to interpret model decisions and provide transparency.
- Best Model Selection: The top-performing model trained on unscaled data was chosen for deployment.
- and...  
##### **Streamlit Web App** where users can:
- Input applicant data via interactive UI
- Receive instant loan prediction (Approved /  Rejected)
- See a LIME visualization explaining the model's reasoning
- Read a simple natural language explanation of key features influencing the result

### How to Run Locally
1. Clone the Repository  
   `git clone https://github.com/Tais19991/Data-Analysis/edit/main/Loan_Approval_Forecast.git`
   `cd Loan_Approval_Forecast`

3. Install Requirements  
     `pip install -r requirements.txt`

5. Run the Streamlit App  
   `streamlit run app/app.py`

7. Using the App  
- A browser window should open automatically.
- If not, open your browser and go to http://localhost:8501
- Enter applicant details and get a real-time loan approval prediction along with a LIME-based explanation.


