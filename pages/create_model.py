import streamlit as st 
from utils.admin_utils import *
from dotenv import load_dotenv
import joblib

#importing keys
load_dotenv()

if 'clean_data' not in st.session_state:
  st.session_state['clean_data']=None
if 'svm_classifier' not in st.session_state:
  st.session_state['svm_classifier']=None
if 'x_test' not in st.session_state:
  st.session_state['x_test']=None
if 'y_test' not in st.session_state:
  st.session_state['y_test']=None



st.header("Let's Build Our Model")
tabs_titles=['Data Preprocessing','Model Training','Model Evaluation','Save Model']
tabs=st.tabs(tabs_titles)

#data preprocessing 
with tabs[0]:
  st.subheader('Data Preprocessing')
  st.write('Here we preprocess the data')
  file=st.file_uploader('Please upload csv file here...',type=['csv'])
  buttton=st.button('Load data',key='data')

  if buttton:
    if file:
      with st.spinner('Loading.......'):
        embed=get_embeddings()
        st.session_state['clean_data']=read_csv_file(file,embed)
      st.success('Preprocessing done')
    else:
      st.error('please upload file')

#model training
with tabs[1]:
  st.subheader('Model Training')
  st.write('Here we train the model')
  buttton=st.button('Train Model',key='train')

  if buttton:
    with st.spinner('Training......'):
      st.session_state['svm_classifier'],st.session_state['x_test'],st.session_state['y_test']=train_model(st.session_state['clean_data'])
    st.success('Model Trained')

#Model Evaluation
with tabs[2]:
  st.subheader('Model Evaluation')
  st.write('Here we evaluate the model')
  buttton=st.button('evaluate Model',key='evaluate')

  if buttton:
    with st.spinner('Evaluating......'):
      st.session_state['score']=get_score(st.session_state['svm_classifier'],st.session_state['x_test'],st.session_state['y_test'])
    st.success('Model Evaluated')
    st.write('Score:')
    st.write(st.session_state['score'])
    query='where is my salary?'
    embed=get_embeddings()
    query_embedd=embed.embed_query(query)
    st.write(query)
    st.write(st.session_state['svm_classifier'].predict([query_embedd])[0])

# Save Model
with tabs[3]:
  st.subheader('Save Model')
  st.write('Here we save the model to use')
  name_model=st.text_input('Enter the name of model')
  buttton=st.button('Save Model',key='save')
  if buttton:
    with st.spinner('Saving......'):
      joblib.dump(st.session_state['svm_classifier'],f'./models/{name_model}.pkl')
    st.success('Model Saved')