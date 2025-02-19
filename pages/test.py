import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = None

@st.cache_resource
def predict_results(test_data, columnsN):
    with open('./models/Scaler.pkl', 'rb') as f:
        scaler = pkl.load(f)
        new_data = scaler.transform(test_data) 
        new_data =  pd.DataFrame(new_data,columns=columnsN) # Apply scaling on the test data
        st.dataframe(new_data, hide_index= True)
        y_pred=model.predict(new_data)
        return y_pred

def take_input(model=None):
    if model is not None:
        with st.form('Select Parameters to test:', ):
            features = pkl.load(open(f'./models/Features.pkl', 'rb'))
            # st.write(features)
            keys = [x for x in features.keys()]
            default_value = None
            new_data = dict.fromkeys(keys, default_value)
                # st.write(param)
            for feature in features.keys():
                options = features[feature].keys()
                # st.write(options)
                tempV1 = features[feature][st.selectbox(f"Select value for {feature[::-1][2:][::-1]}", 
                        options = options) ]
                empL = []
                empL.append(tempV1)
                new_data[feature] = empL
            
            st.session_state['subB'] = st.form_submit_button('Submit')
 
    if 'subB' in st.session_state and st.session_state['subB']:
                # st.write(new_data)
                # st.write(subB) 
        # st.write(pd.DataFrame(new_data).reset_index())
        new_data = pd.DataFrame(new_data)
        columnsN = new_data.columns

        y_pred = predict_results(new_data, columnsN)
        
        if y_pred[0] == 0:
            st.subheader('Predicted Class: Didn\'t Survived')
        else:
            st.subheader('Predicted Class: Survived')    

# @st.cache_resource
def load_model(model, kernel):
    try:
        match model:
            case 'Naive Bayesian':
                st.session_state['model_loaded'] = pkl.load(open(f'./models/NB.pkl', 'rb'))
            case 'Decesion Tree':
                st.session_state['model_loaded'] = pkl.load(open(f'./models/DT.pkl', 'rb'))
            case 'Random Forest':
                st.session_state['model_loaded'] = pkl.load(open(f'./models/RF.pkl', 'rb'))
            case 'SVM':
                match kernel:
                    case 'Linear':
                        st.session_state['model_loaded']= pkl.load(open(f'./models/{model}_linear.pkl', 'rb'))
                    case 'RBF':
                        st.session_state['model_loaded'] = pkl.load(open(f'./models/{model}_rbf.pkl', 'rb'))
                    case 'Polynomial':
                        st.session_state['model_loaded'] = pkl.load(open(f'./models/{model}_poly.pkl', 'rb'))
                    case 'Sigmoidal':
                        st.session_state['model_loaded'] = pkl.load(open(f'./models/{model}_sigmoid.pkl', 'rb'))
        # st.success(st.session_state['model_loaded'])
        # st.success('Model loaded successfully!')
        
        
    except FileNotFoundError:
        st.error('Model not found. Please make sure the model file exists.')
        return None

if st.session_state['model_loaded'] is not None:
    st.subheader(f'Model loaded successfully!: {st.session_state['model_loaded']}')
    take_input(st.session_state['model_loaded'])  
def test_page():
    with st.sidebar:
        st.title('Select Model to test')
        model = st.selectbox('Select Models to test: ', options=['Naive Bayesian', 'Decesion Tree',
                                    'Random Forest', 'SVM'])
        if model == 'SVM':
            kernel = st.selectbox('Select Kernel for SVM: ', options=['Linear', 'RBF', 'Polynomial', 'Sigmoidal'])
        else:
            kernel = 'None'

        b1 = st.button('Load Model', on_click=load_model, args= (model, kernel))

# take_input(st.session_state['model_loaded']) 
test_page()




    