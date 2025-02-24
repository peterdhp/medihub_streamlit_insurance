import streamlit as st
from draft_LLMengine_langgraph03_wotermcon import insurance_wotc_engine
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from menu_streamlit import menu
import datetime




client = Client()

if "user" not in st.session_state:
    st.session_state.user = ''
if "birth" not in st.session_state:
    st.session_state.birth = ''
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if "messages_wo" not in st.session_state:
    st.session_state["messages_wo"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]
if 'log_str' not in st.session_state:
    st.session_state['log_str'] = 'ai: 보험과 관련해서 어떤게 궁금하신가요?'



def reset():
    st.session_state["messages_wo"] = [{"type": "ai", "content": "보험과 관련해서 어떤게 궁금하신가요?"}]
    # st.session_state.thread_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


for msg in st.session_state.messages_wo:
    st.chat_message(msg["type"]).write(msg["content"])

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


#config = {"configurable": {"thread_id": st.session_state.thread_id}}

if prompt := st.chat_input():

    st.session_state.messages_wo.append({"type": "human", "content": prompt})
    st.chat_message("human").write(prompt)
    st.session_state.log_str += 'human: ' + prompt
    with collect_runs() as cb:
        response = insurance_wotc_engine.invoke({"user_input": prompt, "chat_history": st.session_state.messages_wo})
        st.session_state.run_id = cb.traced_runs[-1].id
        #print(st.session_state.run_id)
    if response['type'] == 'not_related' :
        st.session_state.messages_wo.append({"type": "ai", "content": "저는 보험 관련 질문에 대해서만 답변할 수 있어요."})
        st.chat_message("ai").write("저는 보험 관련 질문에 대해서만 답변할 수 있어요.")
        st.session_state.log_str += 'ai: ' + "저는 보험 관련 질문에 대해서만 답변할 수 있어요." + '\n'
    elif response['type'] == 'enroll_info_required' :
        st.session_state.messages_wo.append({"type": "ai", "content": "해당 질문에 답변하기 위해서는 보험 정보 연결이 필요합니다.\n보험 데이터를 연결하시면 좀 더 심층적인 고객님의 현재 상황 파악이 가능합니다."})
        st.chat_message("ai").write("해당 질문에 답변하기 위해서는 보험 정보 연결이 필요합니다.\n보험 데이터를 연결하시면 좀 더 심층적인 고객님의 현재 상황 파악이 가능합니다.")
        st.session_state.log_str += 'ai: ' + "해당 질문에 답변하기 위해서는 보험 정보 연결이 필요합니다.보험 데이터를 연결하시면 좀 더 심층적인 고객님의 현재 상황 파악이 가능합니다." + '\n'
        st.page_link("app_streamlit.py", label="내 보험 연결해 심층 답변 받기", icon=None, help=None, disabled=False, use_container_width=None)
    else :
        msg = response["response"]
        st.session_state.messages_wo.append({"type": "ai", "content": msg})
        st.chat_message("ai").write(msg)
        st.session_state.log_str += 'ai: ' + msg + '\n'
if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    st.text_input('[선택] 코멘트를 입력해주세요.',key="feedback_text")
    st.feedback(
        options="thumbs",
        on_change=submit_feedback,
        key="feedback",
    )
    
with st.sidebar :
    st.button("대화 내역 초기화 하기", on_click=reset)
    st.download_button(
                label="대화내역 저장하기",
                data=st.session_state.log_str,
                file_name=datetime.datetime.now().strftime("%Y%M%D%h%m")+'_보장곰 내역.txt',
                mime='text/plain'
            )
                


menu()
        