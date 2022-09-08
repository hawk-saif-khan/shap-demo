import json
import os

import pandas as pd
import shap
import streamlit as st
from streamlit_shap import st_shap
from xgboost import XGBClassifier


def mains(explainer, feature, sub_data, x_test, encoding, mapping, ROOT_DIR):
    def create_mapping_string(feature_set, encodings, mappings):
        st.write(f"**Encoding for {feature_set} are as follows:** ")
        str = ''
        for i, cat in enumerate(mappings):
            if i == 0:
                str = str + cat + ': ' + encodings[cat]
            else:
                str = str + ', ' + cat + ': ' + encodings[cat]
        return str

    def st_print(feature_set, encodings, mapping):
        if feature_set in mapping.keys():
            categories = mapping[feature_set].split(',')
            string_to_print = create_mapping_string(feature_set, encodings, categories)
            st.write(string_to_print)
        else:
            pass

    def all_details(subset_text):
        st.write("No subsetting was done here.")

    def age_details(subset_text):
        st.write(f"Here we focus on individuals which are **{subset_text}**.")

    def hpw_details(subset_text):
        st.write(f"Here we focus on individuals whose working hours are **{subset_text}**.")

    def categorical_details(subset_text):
        st.write(f"Here we focus on individuals which have the relationship status as **{subset_text}**.")
        if 'education' == subset_text:
            st.write(
                "Education feature is orderly encoded, i.e. the lowest degree is encoded with lowest number (1) "
                "while Doctorate degree is encoded with the highest number in the list.")

    def subsubfocus(text, x, encoding):
        temp_df, c, subset_select = '', '', ''
        if text in ["all", "hours per week", "age"]:
            if "all" in text:
                c = all_details
                temp_df = x
                subset_select = ""
            elif text == "age":
                subset_select = st.sidebar.selectbox("On which range of age do you want to subset", (
                    "age < 25", "25 < age < 35", "35 < age < 45", "45 < age < 60", "60 < age < 70", "age > 70"))
                c = age_details
                ranges = subset_select.split(' < ')
                if len(ranges) == 3:
                    temp_df = x[x["age"] < int(ranges[-1])]
                    temp_df = temp_df[temp_df["age"] > int(ranges[0])]
                elif len(ranges) == 2:
                    temp_df = x[x["age"] < int(ranges[-1])]
                else:
                    ranges = subset_select.split(' > ')
                    temp_df = x[x["age"] < int(ranges[-1])]
            elif text == "hours per week":
                subset_select = st.sidebar.selectbox("On which range of \"hours per week\" do you want to subset?", (
                    "hours < 20", "20 < hours < 35", "35 < hours < 50", "50 < hours < 60", "60 < hours < 80",
                    "hours > 80"))
                c = hpw_details
                ranges = subset_select.split(' < ')
                if len(ranges) == 3:
                    temp_df = x[x["hours_per_week"] < int(ranges[-1])]
                    temp_df = temp_df[temp_df["hours_per_week"] > int(ranges[0])]
                elif len(ranges) == 2:
                    temp_df = x[x["hours_per_week"] < int(ranges[-1])]
                else:
                    ranges = subset_select.split(' > ')
                    temp_df = x[x["hours_per_week"] < int(ranges[-1])]
        # For categorical case
        else:
            categories = mapping[text]
            categories = tuple(categories.split(','))
            subset_select = st.sidebar.selectbox(f"On which {text} do you want to subset?", (categories))
            text = text.replace('-', '_')
            temp_df = x[x[text] == int(encoding[subset_select])]
            c = categorical_details
        return temp_df, c, subset_select

    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(ROOT_DIR, "model/model_sklearn.json"))
    shap_explainer = shap.TreeExplainer(xgb_model)
    df_sub, mapping_text, subset_text = subsubfocus(sub_data, x_test, encoding)
    shap_values = shap_explainer(df_sub)
    if explainer == "Sensitivity":
        st_shap(shap.plots.scatter(shap_values[:, feature], color=shap_values), height=420)
        st.header("Details about the plot:")
        mapping_text(subset_text)
        st.write("The scatter plot points are colored by the feature that seems to have the strongest "
                 "interaction effect with the feature given by the shap")
        st.write("**Understanding the scatter plot:**")
        st.markdown(
            f"- Here the higher y value means that the property \"{feature_select}"
            f"\" played significant role in increasing the salary of the individual above 50K.")
        st.markdown(f"- Whereas the lower y value means that the property \"{feature_select}"
                    f"\" took part in lowering the salary of the individual.")
        st_print(feature_select, encoding, mapping)
    else:
        st_shap(shap.plots.beeswarm(shap_values), height=420)
        st.header("Details about the plot:")
        mapping_text(subset_text)
        st.write("**Understanding the Shap plot:**")
        st.markdown(
            f"- Here the higher x value means the feature value tried its best to increase the salary of the individual"
            f" \"{subset_text}, and vice versa for lower x values")
        st.markdown(f"- The redder the feature means the higher the feature value. "
                    f"(Note that categorical features have been encoded to represent as integer.)")
        st.markdown(f"- This means that if the redder value is on the higher x-axis, higher feature value increases "
                    f"the salary in this case")


@st.cache
def load_data():
    root = '/home/saif/Main/mystuff/Python_projs/shap-demo/output'
    encoding = json.load(open(root + '/dataset/encodings.json'))
    x_test = pd.read_csv(root + "/dataset/x_test.csv")
    return x_test, encoding


x_test, encoding = load_data()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/output/'
mapping_path = os.path.join(ROOT_DIR, 'dataset/mapping.json')
mapping = json.load(open(mapping_path))
feature_names = x_test.columns.tolist()
st.sidebar.title('Salary analysis')
select_explainer = st.sidebar.radio("Choose your desired explainer.", ("Sensitivity", "Shap"))
# encoding = json.load(open(root + '/dataset/encodings.json'))

feature_select = ''
if select_explainer == "Sensitivity":
    feature_select = st.sidebar.selectbox("Select desired feature to analyze",
                                          ("sex", "age", "hours per week", "education", "relationship", "occupation",
                                           "workclass", "marital_status",
                                           "race", "native_country"))
select_subset_dataset = st.sidebar.selectbox(
    "How would you like to focus on the data?",
    ("all", "sex", "age", "hours per week", "education", "relationship", "occupation", "workclass", "marital_status",
     "race", "native_country")
)
with st.spinner('Wait for it...'):
    st.sidebar.button("Refresh",
                      on_click=mains(select_explainer, feature_select, select_subset_dataset, x_test, encoding,
                                     mapping, ROOT_DIR))
