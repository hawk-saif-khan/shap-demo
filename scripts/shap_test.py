import json

import pandas as pd
import shap
import streamlit as st
from streamlit_shap import st_shap
from xgboost import XGBClassifier


def mains(explainer, feature, sub_data, x_test, encoding):
    def st_print(feature_set, encodings):
        if feature_set == "workclass":
            st.write(f"**Encoding for {feature_set} are as follows:** ")
            st.write(
                f'Private: {encodings["Private"]}, Self-emp-not-inc: {encodings["Self-emp-not-inc"]}, Self-emp-inc: '
                f'{encodings["Self-emp-inc"]}, Federal-gov: {encodings["Federal-gov"]}, Local-gov: {encodings["Local-gov"]}'
                f', State-gov: {encodings["State-gov"]}, Without-pay: {encodings["Without-pay"]}'
            )

        elif feature_set == "education":
            st.write(
                "Education feature is orderly encoded, i.e. the lowest degree is encoded with lowest number (1) "
                "while Doctorate degree is encoded with the highest number in the list.")
            st.write(f"**Encoding for {feature_set} are as follows:** ")
            st.write(
                f'Bachelors: {encodings["Bachelors"]}, Some-college: {encodings["Some-college"]}, 11th: {encodings["11th"]}'
                f', HS-grad: {encodings["HS-grad"]}, Prof-school: {encodings["Prof-school"]}'
                f', Assoc-acdm: {encodings["Assoc-acdm"]}, Assoc-voc: {encodings["Assoc-voc"]}'
                f', 9th: {encodings["9th"]}, 7th-8th: {encodings["7th-8th"]}, 12th: {encodings["12th"]}'
                f', Masters: {encodings["Masters"]}, 1st-4th: {encodings["1st-4th"]}, 10th: {encodings["10th"]}'
                f', Doctorate: {encodings["Doctorate"]}, 5th-6th: {encodings["5th-6th"]}'
                f', Preschool: {encodings["Preschool"]}')

        elif feature_set == "marital_status":
            st.write(f"**Encoding for {feature_set} are as follows:** ")
            st.write(f'Married-civ-spouse: {encodings["Married-civ-spouse"]}, Divorced: {encodings["Divorced"]}'
                     f', Never-married: {encodings["Never-married"]}, Separated: {encodings["Separated"]}'
                     f', Widowed: {encodings["Widowed"]}, Married-spouse-absent: {encodings["Married-spouse-absent"]},'
                     f' Married-AF-spouse: {encodings["Married-AF-spouse"]}')

        elif feature_set == "relationship":
            st.write(f"**Encoding for {feature_set} are as follows:** ")
            st.write(f'Wife: {encodings["Wife"]}, Own-child: {encodings["Own-child"]}, Husband: {encodings["Husband"]}'
                     f', Not-in-family: {encodings["Not-in-family"]}, Other-relative: {encodings["Other-relative"]}'
                     f', Unmarried: {encodings["Unmarried"]}')
        elif feature_set == 'sex':
            st.write(f"**Encoding for {feature_set} are as follows:** ")
            st.write(f'Male: {encodings["Male"]}, Female: {encodings["Female"]}')
        else:
            pass

    def all_details(subset_text):
        st.write("No subsetting was done here.")

    def gend_details(subset_text):
        st.write("Here we focus on " + subset_text + " individuals. The mapping is as follows:")
        st.markdown("- Male: " + encoding["Male"])
        st.markdown("- Female: " + encoding["Female"])

    def age_details(subset_text):
        st.write("Here we focus on individuals which are " + subset_text)

    def hpw_details(subset_text):
        st.write("Here we focus on individuals whose working hours are " + subset_text)

    def education_details(subset_text):
        st.write("Here we focus on individuals who have studies till " + subset_text)
        st.write(
            "Education feature is orderly encoded, i.e. the lowest degree is encoded with lowest number (1) "
            "while Doctorate degree is encoded with the highest number in the list.")

    def relationship_details(subset_text):
        st.write("Here we focus on individuals which have the relationship status as " + subset_text)

    def marital_details(subset_text):
        st.write("Here we focus on individuals which have their marital status status as " + subset_text)

    def occupation_details(subset_text):
        st.write("Here we focus on individuals who work as " + subset_text)

    def workclass_details(subset_text):
        st.write("Here we focus on individuals who have their workclass as " + subset_text)

    # def sensitivity_text(feature_name):

    def subsubfocus(text, x, encoding):
        temp_df, c, subset_select = '', '', ''
        if "All" in text:
            c = all_details
            temp_df = x
            subset_select = ""
        if text == "Gender":
            c = gend_details
            subset_select = st.sidebar.selectbox("On which gender do you want to subset", ("Male", "Female"))
            temp_df = x[x["sex"] == int(encoding[subset_select])]
        elif text == "Age":
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
        elif text == "Hours per week":
            subset_select = st.sidebar.selectbox("On which range of \"hours per week\" do you want to subset?", (
                "hours < 20", "20 < hours < 35", "35 < hours < 50", "50 < hours < 60", "60 < hours < 80", "hours > 80"))
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
        elif text == "Education":
            subset_select = st.sidebar.selectbox("On which \"education\" do you want to subset?", (
                "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"))
            temp_df = x[x["education"] == int(encoding[subset_select])]
            c = education_details
        elif text == "Education":
            subset_select = st.sidebar.selectbox("On which \"education\" do you want to subset?", (
                "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"))
            temp_df = x[x["education"] == int(encoding[subset_select])]
            c = education_details
        elif text == "Relationship":
            subset_select = st.sidebar.selectbox("On which \"relationship\" do you want to subset?", (
                "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"))
            temp_df = x[x["relationship"] == int(encoding[subset_select])]
            c = relationship_details
        elif text == "Marital-status":
            subset_select = st.sidebar.selectbox("On which \"marital-status\" do you want to subset?", (
                "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
                "Married-AF-spouse"))
            temp_df = x[x["marital_status"] == int(encoding[subset_select])]
            c = marital_details
        elif text == "Occupation":
            subset_select = st.sidebar.selectbox("On which \"occupation\" do you want to subset?", (
                "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                "Priv-house-serv", "Protective-serv", "Armed-Forces"))
            temp_df = x[x["occupation"] == int(encoding[subset_select])]
            c = occupation_details
        elif text == "Workclass":
            subset_select = st.sidebar.selectbox("On which \"workclass\" do you want to subset?", (
                "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
                "Never-worked"))
            temp_df = x[x["workclass"] == int(encoding[subset_select])]
            c = workclass_details
        return temp_df, c, subset_select

    root = '/home/saif/Main/mystuff/Python_projs/Shapley/output'
    xgb_model = XGBClassifier()
    xgb_model.load_model(root + "/model/model_sklearn.json")
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
        st_print(feature_select, encoding)
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
    root = '/home/saif/Main/mystuff/Python_projs/Shapley/output'
    encoding = json.load(open(root + '/dataset/encodings.json'))
    x_test = pd.read_csv(root + "/dataset/x_test.csv")
    # xgb_model = XGBClassifier()
    # xgb_model.load_model(root + "/model/model_sklearn.json")
    return x_test, encoding


@st.cache
def loaf_model():
    root = '/home/saif/Main/mystuff/Python_projs/Shapley/output'
    xgb_model = XGBClassifier()
    xgb_model.load_model(root + "/model/model_sklearn.json")
    return xgb_model


x_test, encoding = load_data()
feature_names = x_test.columns.tolist()
st.sidebar.title('Salary analysis')
select_explainer = st.sidebar.radio("Choose your desired explainer.", ("Sensitivity", "Shap"))
# encoding = json.load(open(root + '/dataset/encodings.json'))

feature_select = ''
if select_explainer == "Sensitivity":
    feature_select = st.sidebar.selectbox("Select desired feature to analyze",
                                          ("sex", "age", "education", "relationship", "marital_status",
                                           "hours_per_week", "workclass"))
    # st_print(feature_select, encodings=encoding)
select_subset_dataset = st.sidebar.selectbox(
    "How would you like to focus on the data?",
    ("All", "Gender", "Age", "Hours per week", "Education", "Relationship", "Occupation", "Workclass", "Marital-status")
)
with st.spinner('Wait for it...'):
    # time.sleep(5)
    st.sidebar.button("Refresh",
                      on_click=mains(select_explainer, feature_select, select_subset_dataset, x_test, encoding))
