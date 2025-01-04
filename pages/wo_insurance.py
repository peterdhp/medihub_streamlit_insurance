import streamlit as st
from draft_LLMengine_langgraph03_wotermcon import insurance_wotc_engine


insurance_enrollment_info = st.secrets['INSURANCE_ENROLLMENT'][st.session_state.user]

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["type"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"type": "human", "content": prompt})
    st.chat_message("human").write(prompt)
    response = insurance_wotc_engine.invoke({"user_input": prompt, "insurance_enrollment_info" : insurance_enrollment_info, "chat_history":st.session_state.messages})
    if response['type'] == 'not_related' :
        st.session_state.messages.append({"type": "ai", "content": "저는 보험 관련 질문에 대해서만 답변할 수 있어요."})
        st.chat_message("ai").write("저는 보험 관련 질문에 대해서만 답변할 수 있어요.")
    elif response['type'] == 'enroll_info_required' :
        st.session_state.messages.append({"type": "ai", "content": "해당 질문에 답변하기 위해서는 보험 정보 연결이 필요합니다.\n보험 데이터를 연결하시면 좀 더 심층적인 고객님의 현재 상황 파악이 가능합니다."})
        st.chat_message("ai").write("해당 질문에 답변하기 위해서는 보험 정보 연결이 필요합니다.\n보험 데이터를 연결하시면 좀 더 심층적인 고객님의 현재 상황 파악이 가능합니다.")
        st.page_link("pages/wo_insurance.py", label="내 보험 연결해 심층 답변 받기", icon=None, help=None, disabled=False, use_container_width=None)
    else :
        msg = response["response"]
        st.session_state.messages.append({"type": "ai", "content": msg})
        st.chat_message("ai").write(msg)
        
        