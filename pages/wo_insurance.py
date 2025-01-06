import streamlit as st
from draft_LLMengine_langgraph03_wotermcon import insurance_wotc_engine
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from menu_streamlit import menu

menu()

client = Client()

    
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["type"]).write(msg["content"])

def submit_feedback():
    client.create_feedback(
        run_id,
        "thumbs",
        score=feedback,
        comment=feedback_text,
    )
    


if prompt := st.chat_input():

    st.session_state.messages.append({"type": "human", "content": prompt})
    st.chat_message("human").write(prompt)
    with collect_runs() as cb:
        response = insurance_wotc_engine.invoke({"user_input": prompt, "chat_history":st.session_state.messages})
        run_id = cb.traced_runs[0].id
    if response['type'] == 'not_related' :
        st.session_state.messages.append({"type": "ai", "content": "저는 보험 관련 질문에 대해서만 답변할 수 있어요."})
        st.chat_message("ai").write("저는 보험 관련 질문에 대해서만 답변할 수 있어요.")
    elif response['type'] == 'enroll_info_required' :
        st.session_state.messages.append({"type": "ai", "content": "해당 질문에 답변하기 위해서는 보험 정보 연결이 필요합니다.\n보험 데이터를 연결하시면 좀 더 심층적인 고객님의 현재 상황 파악이 가능합니다."})
        st.chat_message("ai").write("해당 질문에 답변하기 위해서는 보험 정보 연결이 필요합니다.\n보험 데이터를 연결하시면 좀 더 심층적인 고객님의 현재 상황 파악이 가능합니다.")
        
        st.page_link("app_streamlit.py", label="내 보험 연결해 심층 답변 받기", icon=None, help=None, disabled=False, use_container_width=None)
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
        
        