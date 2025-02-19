# !pip install --upgrade pip
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from pages.test import test_page
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.title("Training")

def cata2lbl(df, features):
    feature_dict = {}
    for feature in features:
        values = sorted(list(df[feature].unique())) #
        for value in values:
            fea_key = value.upper() if type(value) == str else int(value)
            if feature+'_l' not in feature_dict.keys():
                feature_dict[feature+'_l'] = {fea_key : values.index(value)}
            else:
                feature_dict[feature+'_l'].update({fea_key : values.index(value)})
            df.loc[(df[feature] == value), feature+'_l'] = values.index(value)
        df = df.drop(columns = [feature])
    with open('./models/Features.pkl', 'wb') as f:
        pkl.dump(feature_dict, f)
    return df

def prepare_data(train):
    # Y_train = train['Survived']

    X_train = train.drop(columns = ['Survived'])

    obj_features = X_train.columns
    # st.write("Prepare_data")
    X_train = cata2lbl(X_train, obj_features,)

    return X_train


def col_drop_list(train):
    columns_del = ['PassengerId', 'Name','SibSp', 'Parch', 'Ticket', 'Fare','Cabin', ]
    train = train.drop(columns = columns_del)
    train = prepare_data(train)
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    st.dataframe(train.head(), hide_index = True)    
    return train

def age_band(df):
    df.loc[df['Age'] <= 16, 'AgeBand'] = '0 - 16'
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'AgeBand'] = '17 - 32'
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'AgeBand'] =   '33 - 48'
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'AgeBand'] = '49 - 64'
    df.loc[(df['Age'] > 64),'AgeBand'] = '65 Above'
    return df

# @st.cache_data
def load_data():


    train = pd.read_csv("./dataset/train.csv")
    from sklearn.impute import KNNImputer

    # Assuming train_data is your DataFrame
    imputer = KNNImputer(n_neighbors=5)

    # Create a DataFrame with 'Age' as a 2D array
    age_2d = train[['Age']]

    # Impute the missing values
    age_imputed = imputer.fit_transform(age_2d)

    # Replace the 'Age' column with the imputed values
    train['Age'] = age_imputed

    train = train.dropna()
    train = age_band(train)
    train = train.drop(columns = ['Age'])
    # col_drop_list(train)
    X_train = col_drop_list(train)
    # X_train, Y_train = prepare_data(train)
    return X_train, train['Survived']




# le = LabelEncoder()

# X_trainC = X_train.select_dtypes(include=[object]).apply(le.fit_transform)
# X_trainC['AgeB'] = X_train['AgeBand']

# X_testC = test.select_dtypes(include=[object]).apply(le.fit_transform)
# X_testC['AgeB'] = test['AgeBand']

# st.dataframe(X_trainC,hide_index=True)
# st.dataframe(X_testC,hide_index=True)

res_comp = dict.fromkeys(['Model', 'Kernel', 'Score'])
modelD = list()
kernelD = list()
scoreD = list()
pModel = st.button("Prepare Model",)
if pModel == True:
    X_train, Y_train = load_data()

    st.dataframe(X_train.head(), hide_index=True)

    # result_comp = pd.DataFrame(columns= ['Model', 'Kernel', 'Score'])

    # modelD = list()
    # kernelD = list()
    # scoreD = list()

    st.header('Data Loading completed')

    st.header("Model train")

    models = {
        "SVM" : {'model' : SVC(), 'kernel' : ['linear', 'rbf', 'poly', 'sigmoid'] },
        "DT" : DecisionTreeClassifier(),
        "RF" : RandomForestClassifier(),
        "NB" : GaussianNB()
    }
# #  

    # st.dataframe(X_train,)
    model_name = list(models.keys())
    # st.write(f"{model_name}:")
    for model in model_name:
        if model == "SVM":
            scaler = StandardScaler()
            for kernel in models[model]['kernel']:
                clf = models[model]['model'].set_params(kernel=kernel)
                # X_train = scaler.fit_transform(X_train)
                clf.fit(X_train, Y_train)
                score = cross_val_score(clf, X_train, Y_train, cv=5).mean()

                modelD.append(model)
                kernelD.append(kernel)
                scoreD.append(score)

                # res_comp.update({'Model': model, 'Kernel': kernel, 'Score': score},)
                # st.write(f"{model} : {kernel}: {score}")
                with open(f'./models/{model}_{kernel}.pkl', "wb") as f:
                    pkl.dump(clf, f)

        else:
            clf = models[model].fit(X_train, Y_train)
            score = cross_val_score(clf, X_train, Y_train, cv=5).mean()

            modelD.append(model)
            # kernel = kernel.append(kernel)
            kernelD.append(None)
            scoreD.append(score)

            # res_comp.update({'Model': model, 'Kernel': None, 'Score': score},)
            # st.write(f"{model}: {score}")
            with open(f'./models/{model}.pkl', "wb") as f:
                pkl.dump(clf, f)

    res_comp['Model'] = modelD
    res_comp['Kernel'] = kernelD
    res_comp['Score'] = scoreD

    st.dataframe(pd.DataFrame(res_comp), hide_index=True)

# st.button("Test Model", on_click=test_page)

