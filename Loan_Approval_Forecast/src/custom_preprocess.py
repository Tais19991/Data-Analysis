from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import numpy as np
import pandas as pd


class NegativeDataCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer that detects and handles negative values in a DataFrame.

    Negative values are assumed to be invalid and are replaced with the 
    column median. A warning is raised for each column where negative 
    values are found.
    """
    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        X = X.copy()
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            if (X[col] < 0).sum() > 0:
                warnings.warn(f"Negative values detected in '{col}'. They will be replaced with the column median.", UserWarning)
                X.loc[X[col] < 0, col] = np.nan
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        
        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns_


class NewFeaturesCreator(BaseEstimator, TransformerMixin):
    """
    Transformer that creates domain-specific engineered features 
    from an input DataFrame for loan default prediction tasks.

    New features created:
    - `all_assets`: Sum of residential, commercial, bank, and luxury asset values.
    - `income_per_person`: Income per household member (income / (dependents + 1)).
    - `debt_income_ratio`: Loan amount divided by annual income.
    - `debt_assets_ratio`: Loan amount divided by total assets.
    - `amount_term_ratio`: Loan amount divided by loan term.
    """
    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        return self  

    def transform(self, X, y=None):
        X = X.copy()

        # Dynamically locate required columns using regex
        asset_cols = X.filter(regex='assets_value').columns.tolist()
        income_col = X.filter(regex='income_annum').columns[0]
        loan_col = X.filter(regex='loan_amount').columns[0]
        term_col = X.filter(regex='loan_term').columns[0]
        dependents_col = X.filter(regex='no_of_dependents').columns[0]

        # Create new features
        X['all_assets'] = X[asset_cols].sum(axis=1)
        X['income_per_person'] = X[income_col] / (X[dependents_col] + 1)
        X['debt_income_ratio'] = X[loan_col] / (X[income_col] + 1)
        X['debt_assets_ratio'] = X[loan_col] / (X['all_assets'] + 1)
        X['amount_term_ratio'] = X[loan_col] / (X[term_col] + 1)

        self.columns_ = X.columns
        return X  

    def get_feature_names_out(self, input_features=None):
        return self.columns_