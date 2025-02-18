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
# if 'columns_del' in st.session_state:
#     del st.session_state['columns_del']


def cata2lbl(df, features):
    feature_dict = {}
    for feature in features:
        values = list(df[feature].unique())
        for value in values:
            # st.write(feature_dict[feature] )
            if feature+'_l' not in feature_dict.keys():
                feature_dict[feature+'_l'] = {value.upper(): values.index(value)}
            else:
                feature_dict[feature+'_l'].update({value.upper(): values.index(value)})
            df.loc[(df[feature] == value), feature+'_l'] = values.index(value)
        df = df.drop(columns = [feature])
    with open('./models/Features.pkl', 'wb') as f:
        pkl.dump(feature_dict, f)
    return df

def prepare_data(train):
    Y_train = train['Survived']

    X_train = train.drop(columns = ['Survived'])
    # st.write("Prepare_data")
    # st.dataframe(train, hide_index=True)
    obj_features = list(X_train.select_dtypes(include=[object]))
    # st.write("Prepare_data")
  
    train = cata2lbl(train, obj_features,)

    #X_train = X_train.drop(columns = obj_features)
    
    # with st.expander('Data: '):
    #     st.dataframe(train,hide_index=True)
    #     st.dataframe(Y_train,hide_index=True)
    return X_train, Y_train

def col_drop_list(train):

    with st.sidebar:
        st.title("Select Features to delete: ")
        with st.form('select features from delete:', clear_on_submit=True):
            columns_del = st.multiselect("Select Columns: ", 
                            ['PassengerId', 'Pclass', 'Name','SibSp', 'Parch', 'Ticket', 'Fare','Cabin', ], default=None)
            update = st.form_submit_button("Delete Columns",)
            if update:
                train = train.drop(columns = columns_del)

    train, Y_train = prepare_data(train)
    
    return train, Y_train


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
    X_train, Y_train = col_drop_list(train)
    # X_train, Y_train = prepare_data(train)
    return X_train, Y_train




# le = LabelEncoder()

# X_trainC = X_train.select_dtypes(include=[object]).apply(le.fit_transform)
# X_trainC['AgeB'] = X_train['AgeBand']

# X_testC = test.select_dtypes(include=[object]).apply(le.fit_transform)
# X_testC['AgeB'] = test['AgeBand']

# st.dataframe(X_trainC,hide_index=True)
# st.dataframe(X_testC,hide_index=True)
st.session_state['X_train'], st.session_state['Y_train'] = load_data()
st.header('Data Loading completed')

with st.expander('Training Data: '):
        st.dataframe(X_train,hide_index=True)

st.header("Model train")

pModel = st.button("Preapare Model",)
if pModel == True:
    models = {
        "SVM" : {'model' : SVC(), 'kernel' : ['linear', 'rbf', 'poly', 'sigmoid'] },
        "DT" : DecisionTreeClassifier(),
        "RF" : RandomForestClassifier(),
        "NB" : GaussianNB()
    }
# #  
    X_train, Y_train = st.session_state['X_train'], st.session_state['Y_train']

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
                st.write(f"{model} : {kernel}: {score}")
                with open(f'./models/{model}_{kernel}.pkl', "wb") as f:
                    pkl.dump(clf, f)

        else:
            clf = models[model].fit(X_train, Y_train)
            score = cross_val_score(clf, X_train, Y_train, cv=5).mean()
            st.write(f"{model}: {score}")
            with open(f'./models/{model}.pkl', "wb") as f:
                pkl.dump(clf, f)


# st.button("Test Model", on_click=test_page)

