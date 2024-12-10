# libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# the custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class that we are going to use from here on to predict new data
class loan_model():

    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files which were saved
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            self.df_with_predictions = None

    def load_and_clean_data(self, data_file):
        # import the data
        df = pd.read_csv(data_file, delimiter=',', index_col=0)
        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()
        df.columns = df.columns.str.replace(' ', '')

        # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
        df['loan_status'] = 'NaN'

        # Preprocess data
        df['education'] = df['education'].map({' Not Graduate': 0, ' Graduate': 1})
        df['self_employed'] = df['self_employed'].map({' No': 0, ' Yes': 1})
        df['loan_status'] = df['loan_status'].map({' Rejected': 0, ' Approved': 1})
        df.loc[df['residential_assets_value'] < 0, 'residential_assets_value'] = 0

        #  additional variables which can potentially decrease multicollinearity
        df['all_assets'] = df['residential_assets_value'] + df['commercial_assets_value'] + df['bank_asset_value'] + df[
            'luxury_assets_value']
        df['income_per_person'] = df['income_annum'] / (df['no_of_dependents'] + 1)
        df['debt_income_ratio'] = df['loan_amount'] / df['income_annum']
        df['debt_assets_ratio'] = df['loan_amount'] / df['all_assets']

        # re-order the columns in df
        column_names_reordered = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                                  'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                                  'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value',
                                  'all_assets', 'income_per_person', 'debt_income_ratio',
                                  'debt_assets_ratio', 'loan_status']
        df = df[column_names_reordered]

        # to avoid multicollinearity, drop  columns from df
        columns_to_drop = ['loan_amount', 'income_annum', 'residential_assets_value',
                           'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
        df = df.drop(columns_to_drop, axis=1)

        # drop the variables we decide we don't need for regr
        df = df.drop(['no_of_dependents', 'education', 'self_employed'], axis=1)

        # we have included this line of code if you want to call the 'preprocessed data'
        self.preprocessed_data = df.copy()

        # we need this line so we can use it in the next functions
        self.data = self.scaler.transform(df)

    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    def predicted_outputs(self):
        """predict the outputs and the probabilities and add columns with these values at the end of the new data"""
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data
