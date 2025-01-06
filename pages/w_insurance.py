import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser, NumberedListOutputParser, CommaSeparatedListOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.tracers.context import collect_runs
import os
from operator import itemgetter
from langsmith import traceable
from draft_LLMengine_langgraph03_termcon import insurance_engine

from langsmith import Client
from menu_streamlit import menu_with_redirect

menu_with_redirect()
client = Client()
def submit_feedback():
    client.create_feedback(
        run_id,
        "thumbs",
        score=feedback,
        comment=feedback_text,
    )

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
    with collect_runs() as cb:
        response = insurance_engine.invoke({"user_input": prompt, "insurance_enrollment_info" : insurance_enrollment_info, "chat_history":st.session_state.messages})
        run_id = cb.traced_runs[0].id
    if response['non_related'] == 'F' :
        st.session_state.messages.append({"type": "ai", "content": "저는 건강보험 관련 질문에 대해서만 답변할 수 있어요."})
        st.chat_message("ai").write("저는 건강보험 관련 질문에 대해서만 답변할 수 있어요.")
    else :
        msg = response["response"]
        st.session_state.messages.append({"type": "ai", "content": msg})
        st.chat_message("ai").write(msg)
    if run_id:
        feedback_text = st.text_input("피드백을 입력해주세요")
        feedback = st.feedback(
            options="thumbs",
            on_change = submit_feedback(),
        )
    
        
        if response['end_of_session'] != 'continue' :
            st.session_state.messages.append({"type": "ai", "content": "궁금한 점이 잘 해소되었나요? \n 더 질문을 하셔도 좋고 상담 요약을 전달드릴 수 있어요.\n [end_of_session token] : " + response['end_of_session']})
            st.chat_message("ai").write("궁금한 점이 잘 해소되었나요? \n 더 질문을 하셔도 좋고 상담 요약을 전달드릴 수 있어요.\n [end_of_session token] : " + response['end_of_session'])
            
            
