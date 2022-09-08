import os

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
plt.style.use('seaborn-whitegrid')


# dataset = pd.read_csv("./da")


def prepare(input_path, test_ratio=25, random_state=42, balanced=False, one_hot=False):
    test_size = int(test_ratio) / 100
    # validation_size = int(val_ratio) / 100
    df = pd.read_csv(input_path,
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                            'hours_per_week', 'native_country', 'target'])
    df = df.drop(['fnlwgt'], axis=1)
    df = df.drop(['capital_gain'], axis=1)
    df = df.drop(['capital_loss'], axis=1)
    df = df[(df != ' ?').all(1)]
    # df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
    # df['education_num'] = (df['education_num'] - df['education_num'].mean()) / df['education_num'].std()
    # df['capital_gain'] = (df['capital_gain'] - df['capital_gain'].mean()) / df['capital_gain'].std()
    # df['capital_loss'] = (df['capital_loss'] - df['capital_loss'].mean()) / df['capital_loss'].std()
    # df['hours_per_week'] = (df['hours_per_week'] - df['hours_per_week'].mean()) / df['hours_per_week'].std()
    class1 = df[df['target'].str.contains('>')]
    if balanced:
        class0 = df[df['target'].str.contains('<')].sample(class1.shape[0])
    else:
        class0 = df[df['target'].str.contains('<')]
    class1.target = 1
    class0.target = 0
    df = pd.concat([class0, class1])
    if one_hot:
        native_country = pd.get_dummies(df['native_country'], prefix='country')
        marital_status = pd.get_dummies(df['marital_status'], prefix='marital_status')
        occupation = pd.get_dummies(df['occupation'], prefix='occupation')
        relationship = pd.get_dummies(df['relationship'], prefix='relationship')
        race = pd.get_dummies(df['race'], prefix='race')
        workclass = pd.get_dummies(df['workclass'], prefix='workclass')
        sex = pd.get_dummies(df['sex'], prefix='sex')
        df = df.drop(['workclass', 'marital_status',
                      'occupation', 'relationship', 'race', 'sex', 'native_country'], axis=1)
        df = pd.concat([df, native_country], axis=1)
        df = pd.concat([df, workclass], axis=1)
        df = pd.concat([df, marital_status], axis=1)
        df = pd.concat([df, occupation], axis=1)
        df = pd.concat([df, relationship], axis=1)
        df = pd.concat([df, race], axis=1)
        df = pd.concat([df, sex], axis=1)
        categorical_cols = ['education']
    else:
        categorical_cols = ['workclass', 'marital_status',
                            'occupation', 'relationship', 'race', 'native_country']
    encoding = LabelEncoder()
    encoding.fit(df[categorical_cols].stack().unique())
    for i in categorical_cols:
        df[i] = encoding.transform(df[i])
    en1 = encoding.classes_.tolist()
    en2 = encoding.transform(encoding.classes_).tolist()
    path = '../output/dataset/'
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    # encoded[0]
    with open(path + '/encodings.json', 'w') as file:
        for i, value in enumerate(en1):
            if i == 0:
                file.write('{\n\"' + str(en1[i]) + '\": ' + str(en2[i]) + ',\n')
            elif i == len(en1) - 1:
                file.write('\"' + str(en1[i]) + '\": ' + str(en2[i]) + '\n}')
            else:
                file.write('\"' + str(en1[i]) + '\": ' + str(en2[i]) + ',\n')
    educations = {
        'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 5, '9th': 6, '10th': 7, '11th': 8, '12th': 9,
        'HS-grad': 10, 'Prof-school': 11, 'Assoc-acdm': 12, 'Assoc-voc': 13, 'Some-college': 14, 'Bachelors': 15,
        'Masters': 16, 'Doctorate': 17
    }
    sex = {"Male": 0,
           "Female": 1}
    x = df.drop("target", axis=1)
    # x.education[x.education == 'Bachelors'] = educations
    print(educations['Bachelors'])
    for i in educations:
        x = x.replace(" " + i, educations[i])
    for i in sex:
        x = x.replace(" " + i, sex[i])
    y = df.target
    # x = x.drop(columns=['capital_gain', 'capital_loss', 'native_country', 'race', 'relationship'])
    # x = x.drop(columns=['capital_gain', 'capital_loss'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=int(random_state),
                                                        stratify=y)
    # with tempfile.TemporaryDirectory() as dirpath:
    x_train.to_csv(path + '/x_train.csv', header=True, index=False)
    y_train.to_csv(path + '/y_train.csv', header=True, index=False)
    x_test.to_csv(path + '/x_test.csv', header=True, index=False)
    y_test.to_csv(path + '/y_test.csv', header=True, index=False)


def train(save_model=False):
    x_train = pd.read_csv("../output/dataset/x_train.csv")
    y_train = pd.read_csv("../output/dataset/y_train.csv")
    x_test = pd.read_csv("../output/dataset/x_test.csv")
    y_test = pd.read_csv("../output/dataset/y_test.csv")

    if save_model:
        xgb_model = XGBClassifier(n_estimators=400, learning_rate=0.05).fit(x_train, y_train)
        os.makedirs('../output/model/', exist_ok=True)
        xgb_model.save_model("../output/model/model_sklearn.json")
    # results = xgb_model.predict(val_X)
    # print("Accuracy: %s%%" % (100 * accuracy_score(val_y, results)))
    # print(classification_report(val_y, results))


# prepare("../dataset/temp.csv")
train(save_model=True)
