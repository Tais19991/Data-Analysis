import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pcl
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from custom_preprocess import NewFeaturesCreator, NegativeDataCleaner
from lime.lime_tabular import LimeTabularExplainer

# == Model ==
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model_unscaled_data.pkl'))
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

with open(model_path, "rb") as f:
    model = pcl.load(f)

# == Pipeline ==
pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipelines',
                                             'prep_pipeline_v2_non_scaled.pkl'))
if not os.path.exists(pipeline_path):
    st.error(f"Pipeline not found: {pipeline_path}")
    st.stop()

with open(pipeline_path, 'rb') as p:
    pipeline = pcl.load(p)

# == Train DF for Lime ===
df_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_transformed_non_scaled.csv')
df_path = os.path.abspath(df_path)
df = pd.read_csv(df_path, index_col=0)


# == Text for Lime explanation ==
def generate_lime_explanation_text(class_label: str, lime_weights: list, top_n: int = 10) -> str:
    """
    Generates a user-friendly explanation from LIME output (as_list).
    :return: A formatted string(table) with explanation
    """
    explanation = f"#### Reasons for the decision:\n\n"
    explanation += f"| Feature (scaled) | Effect on  '{class_label}'  decision | Impact on model |\n"
    explanation += "|---------|--------|-------------|\n"

    for feature, weight in lime_weights[:top_n]:
        impact = "🟩 Positive" if weight > 0 else "🟥 Negative"
        condition = feature
        explanation += f"| `{condition}` | {impact} | {abs(weight):.2f} |\n"

    return explanation


# === Streamlit  Input Form ===

st.title("Loan Approval Predictor")
st.markdown("Enter applicant details below:")

dependents = st.selectbox("Dependents", ["0", "1", "2", "3", "4", "5+"])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self-Employed", ['Yes', 'No'])
loan_term = st.selectbox("Loan Term", ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20'])
loan_amount = st.number_input("Loan Amount", min_value=300000, max_value=100000000000)
income_annum = st.number_input("Annual Income", min_value=100000, max_value=100000000000)
cibil_score = st.number_input("CIBIL score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets", min_value=0, max_value=100000000)
commercial_assets_value = st.number_input("Commercial Assets", min_value=0, max_value=100000000)
luxury_assets_value = st.number_input("Luxury Assets", min_value=0, max_value=100000000)
bank_asset_value = st.number_input("Bank Assets", min_value=0, max_value=100000000)

# When Predict Button Clicked
if st.button("Predict Loan Approval"):
    user_data = pd.DataFrame({
        'no_of_dependents': int(dependents),
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': int(loan_term),
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
    }, index=[0])

    prepared_data = pipeline.transform(user_data)
    prediction = model.predict(prepared_data)
    prediction_label = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.write(f"**Loan Status:** {prediction_label}")

    # plot lime graph
    st.subheader(f"{prediction_label}")
    explainer = LimeTabularExplainer(training_data=df.values,
                                     feature_names=[col.removeprefix('num__').removeprefix('cat__') for col in
                                                    df.columns],
                                     class_names=model.classes_.tolist(),
                                     mode='classification'
                                     )

    instance = prepared_data.values[0]
    exp = explainer.explain_instance(instance,
                                     lambda x: model.predict_proba(np.array(x)),
                                     top_labels=1)
    label = exp.top_labels[0]
    list_exp = exp.as_list(label=label)

    try:
        # show lime chart
        plt.figure(figsize=(20, 8))
        fig = exp.as_pyplot_figure(label=exp.top_labels[0])
        plt.title(f"LIME Explanation for applicant,\n "
                  f"prediction - {prediction_label}")
        plt.tight_layout()
        st.pyplot(fig)
        # show chart explanation
        lime_text = generate_lime_explanation_text(f"{prediction_label}", list_exp)
        st.markdown(lime_text)

    except Exception as e:
        st.error("Plotting failed:" + str(e))

    finally:
        print('Thanks for incorporating my model into your decision-making')
