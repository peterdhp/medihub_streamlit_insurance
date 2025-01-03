import streamlit as st
from menu_streamlit import menu
import os

from langsmith import traceable


st.set_page_config(layout="wide")

if "openai_api_key_psscode" not in st.session_state:
    st.session_state.openai_api_key_psscode = ''


os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']
os.environ['OPENAI_API_KEY']=st.secrets['OPENAI_API_KEY']

os.environ['CO_API_KEY']=st.secrets['CO_API_KEY']

st.title("사용자 로그인")

st.session_state.user = st.text_input('이름' )
st.session_state.birth = st.text_input('주민번호 앞 6자리' ,type='password')

user_data = [{'name':'김민수','birth':'123456'},]


    
if {'name' : st.session_state.user,'birth':st.session_state.birth } not in st.secrets['USER_DATA']:
    st.warning('올바르지 않은 값이거나 등록되지 않은 사용자입니다.', icon='⚠')
    
if {'name' : st.session_state.user,'birth':st.session_state.birth } in st.secrets['USER_DATA']:
    st.switch_page('pages/docker_streamlit.py')
    
    

    
menu()
    



