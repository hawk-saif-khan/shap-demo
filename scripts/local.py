import os
from pathlib import Path
import tkinter

import matplotlib
from streamlit_shap import st_shap
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import streamlit as st

st.title('Salary analysis')


def check_model(x_test, y_test, root):
    xgb_model = XGBClassifier()
    xgb_model.load_model(root + "/model/model_sklearn.json")
    results = xgb_model.predict(x_test)
    print("Accuracy: %s%%" % (100 * accuracy_score(y_test, results)))
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(x_test)
    # shap.summary_plot(shap_values, x_test)
    # shap.plots.bar(shap_values.cohorts(2).abs.mean(0))
    # shap.plots.waterfall(shap_values[0])
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # shap.plots.waterfall(shap_values[0])
    # shap.plots.waterfall(shap_values[0])
    st_shap(shap.plots.scatter(shap_values[:, "hours_per_week"]))


# check_model()
# st.text("Fetching data...")
root = '/home/saif/Main/mystuff/Python_projs/Shapley/output'
x_test = pd.read_csv(root + "/dataset/x_test.csv")
y_test = pd.read_csv(root + "/dataset/y_test.csv")
# if st.button('Say hello'):
check_model(x_test, y_test, root)
# else:
#     st.write('Goodbye')
