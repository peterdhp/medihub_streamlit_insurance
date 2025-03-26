import streamlit as st
import pymongo

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


def authenticated_menu():
    st.sidebar.page_link("app_streamlit.py", label="사용자 변경하기")
    st.sidebar.page_link("pages/signup.py", label="보험정보 등록하기")
    st.sidebar.page_link("pages/wo_insurance.py", label="보험관련 질문하기(가입정보 x)")
    st.sidebar.page_link("pages/w_insurance.py", label="보험관련 질문하기(가입정보 ㅇ)")
    #st.sidebar.page_link("pages/w_insurance.py", label="보험관련 질문하기(stream)")
    #st.sidebar.page_link("pages/result_opinion_new.py", label="새로운 검사 소견 자동 작성 및 소견 요약")
    #st.sidebar.page_link("pages/overall_opinion.py", label="종합 소견 자동 작성")
    #st.sidebar.page_link("pages/overall_opinion_english.py", label="종합 소견 자동 작성 (영문버전)")
    
def unauthenticated_menu():
    st.sidebar.page_link("pages/wo_insurance.py", label="보험관련 질문하기(가입정보 x)")
    st.sidebar.page_link("pages/signup.py", label="보험정보 등록하기")
    st.sidebar.page_link("app_streamlit.py", label="로그인하기")


def menu():
    with st.sidebar:
        if not any(item['name'] == st.session_state.user and item['birthdate'] == st.session_state.birth for item in st.session_state.user_data):

        #if {'name' : st.session_state.user,'birth':st.session_state.birth } not in st.session_state.user_data:
            unauthenticated_menu()
            return
    authenticated_menu()

def menu_with_redirect():
    if not any(item['name'] == st.session_state.user and item['birthdate'] == st.session_state.birth for item in st.session_state.user_data):

    #if {'name' : st.session_state.user,'birth':st.session_state.birth } not in st.session_state.user_data:
        st.switch_page("app_streamlit.py")
    menu()