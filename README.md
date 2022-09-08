## About the Dataset
- Classification dataset.
- Predict whether income exceeds $50K/yr based on census data. Also known as "Adult" dataset.
- Features details:
  - age: continuous.
  - workclass: categorical
  - education: categorical
  - education-num: continuous
  - marital-status: continuous
  - occupation: categorical
  - relationship: categorical
  - race: categorical
  - sex: categorical
  - hours-per-week: continuous
  - native-country: categorical

## Pipeline steps.
- **Setup_main.py** is for preprocessing and training the data.
  - Data can be cleaned in multiple ways, (e.g. One hot encoding or labeled encoding, or making the dataset to balance in class) in this pipeline.
  - XGboost is used for training.
  - Obtained accuracy is 83.8%.
  - Necessary file such as train/test.csv, model.json and encoding.json are saved under output/.
- **shap_demo.py** is the streamlit app which can be run using the following command

     `streamlit run scripts/shap_demo.py`