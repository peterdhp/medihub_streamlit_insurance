import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from menu_streamlit import menu_with_redirect



if "messages" not in st.session_state:
    st.session_state.messages = []
    
if st.session_state.add_question == []  :
    st.session_state.add_question = ['']
    
    
def demo():
    st.session_state.status = "chat"
    st.session_state.messages = []


for message in st.session_state.messages:
    role = '🩺' if message[0] == 'ai' else message[0]
    with st.chat_message(role):
        st.markdown(message[1])
        
        

with st.sidebar:
    st.button("Demo",on_click=demo)
menu_with_redirect()
