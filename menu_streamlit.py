import streamlit as st



user_data = [{'name':'박도훈','birth':'medihub'},{'name':'조화윤','birth':'medihub'},{'name':'문성수','birth':'medihub'},{'name':'송지은','birth':'medihub'},{'name':'고준현','birth':'medihub'},{'name':'박근목','birth':'medihub'},{'name':'박현우','birth':'medihub'},{'name':'송원철','birth':'medihub'},{'name':'조민승','birth':'medihub'},{'name':'조송만','birth':'medihub'}]


def authenticated_menu():
    st.sidebar.page_link("app_streamlit.py", label="사용자 변경하기")
    st.sidebar.page_link("pages/wo_insurance.py", label="보험관련 질문하기(가입정보 x)")
    st.sidebar.page_link("pages/w_insurance.py", label="보험관련 질문하기(가입정보 ㅇ)")
    #st.sidebar.page_link("pages/result_opinion_new.py", label="새로운 검사 소견 자동 작성 및 소견 요약")
    #st.sidebar.page_link("pages/overall_opinion.py", label="종합 소견 자동 작성")
    #st.sidebar.page_link("pages/overall_opinion_english.py", label="종합 소견 자동 작성 (영문버전)")
    
def unauthenticated_menu():
    st.sidebar.page_link("pages/wo_insurance.py", label="보험관련 질문하기(가입정보 x)")
    st.sidebar.page_link("app_streamlit.py", label="사용자 등록하기")


def menu():
    with st.sidebar:
        if {'name' : st.session_state.user,'birth':st.session_state.birth } not in user_data:
            unauthenticated_menu()
            return
    authenticated_menu()

def menu_with_redirect():
    if {'name' : st.session_state.user,'birth':st.session_state.birth } not in user_data:
        st.switch_page("app_streamlit.py")
    menu()