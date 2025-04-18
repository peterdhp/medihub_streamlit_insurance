import streamlit as st
from menu_streamlit import menu
import os
import pymongo
from langsmith import traceable
import socket
import logging




if "openai_api_key_psscode" not in st.session_state:
    st.session_state.openai_api_key_psscode = ''


os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']
os.environ['OPENAI_API_KEY']=st.secrets['OPENAI_API_KEY']

os.environ['CO_API_KEY']=st.secrets['CO_API_KEY']




st.title("사용자 로그인")

if "user" not in st.session_state:
    st.session_state.user = ''
if "birth" not in st.session_state:
    st.session_state.birth = ''
st.session_state.user = st.text_input('이름' )
st.session_state.birth = st.text_input('주민번호 앞 6자리(비밀번호)' ,type='password')

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo_connection_string"])


if 'user_data' not in st.session_state:
    client = init_connection()
    db = client['insurance_demo']
    collection = db['users']
    items = collection.find()
    st.session_state.user_data = list(items)
    

#user_data = [{'name':'박도훈','birth':'medihub'},{'name':'조화윤','birth':'medihub'},{'name':'문성수','birth':'medihub'},{'name':'송지은','birth':'medihub'},{'name':'고준현','birth':'medihub'},{'name':'박근목','birth':'medihub'},{'name':'박현우','birth':'medihub'},{'name':'송원철','birth':'medihub'},{'name':'조민승','birth':'medihub'},{'name':'조송만','birth':'medihub'},{'name':'메디허브','birth':'medihub'},{'name':'손준명','birth':'medihub'}]
#print(st.session_state.user_data)

    
if not any(item['name'] == st.session_state.user and item['birthdate'] == st.session_state.birth for item in st.session_state.user_data):
    st.warning('올바르지 않은 값이거나 등록되지 않은 사용자입니다.', icon='⚠')

    
if any(item['name'] == st.session_state.user and item['birthdate'] == st.session_state.birth for item in st.session_state.user_data):
    st.switch_page('pages/w_insurance.py')
    
    
if st.button('보험 정보 등록하기'):
    st.switch_page('pages/signup.py')
    
    





