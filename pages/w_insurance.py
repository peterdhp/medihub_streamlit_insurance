import streamlit as st
from langchain_core.tracers.context import collect_runs
from LLMengine_split_0110 import insurance_engine
from langsmith import Client
from menu_streamlit import menu_with_redirect
import datetime

client = Client()



if "user" not in st.session_state:
    st.session_state.user = ''
if "birth" not in st.session_state:
    st.session_state.birth = ''
#if "thread_id" not in st.session_state:
#    st.session_state.thread_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

if "messages_w" not in st.session_state:
    st.session_state["messages_w"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]
    #st.session_state.thread_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
menu_with_redirect()

def reset():
    st.session_state["messages_w"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]

with st.sidebar :
    st.button("대화 내역 초기화 하기", on_click=reset)


insurance_enrollment_info = st.secrets['INSURANCE_ENROLLMENT'][st.session_state.user]

#config = {"configurable": {"thread_id": st.session_state.thread_id}}

def submit_feedback():
    client.create_feedback(
        run_id,
        "thumbs",
        score=st.session_state.feedback,
        comment=st.session_state.feedback_text
    )
    st.session_state.feedback = None
    st.session_state.feedback_text = None
    st.session_state.run_id = None

for msg in st.session_state.messages_w:
    st.chat_message(msg["type"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages_w.append({"type": "human", "content": prompt})
    st.chat_message("human").write(prompt)
    with collect_runs() as cb:
        with st.spinner('보장곰이 가입하신 보험들을 살펴보고 있습니다.'):
            response = insurance_engine.invoke({"user_input": prompt, "insurance_enrollment_info" : insurance_enrollment_info, "chat_history": st.session_state.messages_w},config={"recursion_limit": 15})
        st.session_state.run_id = cb.traced_runs[-1].id
        #print(st.session_state.run_id)
    if response['non_related'] == 'F' :
        st.session_state.messages_w.append({"type": "ai", "content": "저는 건강보험 관련 질문에 대해서만 답변할 수 있어요."})
        st.chat_message("ai").write("저는 건강보험 관련 질문에 대해서만 답변할 수 있어요.")
    else :
        msg = response["response"]
        details = response.get("report")
        st.session_state.messages_w.append({"type": "ai", "content": msg})
        st.chat_message("ai").write(msg)
        
        if details :
            with st.expander("See explanation"):
                details_str = "\n\n".join(f"{key}: {value}" for key, value in details.items())
                st.write(details_str)
   
    
    if 'end_of_session' in response :    
        if response['end_of_session'] != 'continue' :
            st.session_state.messages_w.append({"type": "ai", "content": "궁금한 점이 잘 해소되었나요? \n 더 질문을 하셔도 좋고 상담 요약을 전달드릴 수 있어요.\n [end_of_session token] : " + response['end_of_session']})
            st.chat_message("ai").write("궁금한 점이 잘 해소되었나요? \n 더 질문을 하셔도 좋고 상담 요약을 전달드릴 수 있어요.\n [end_of_session token] : " + response['end_of_session'])
            
            
if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    st.text_input('[선택]코멘트를 입력해주세요.',key="feedback_text")
    st.feedback(
        options="thumbs",
        on_change=submit_feedback,
        key="feedback",
    )
            
