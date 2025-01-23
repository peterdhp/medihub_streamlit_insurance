import os
import time
import gzip
import json
import pickle
import datetime

import pymupdf4llm
import pandas as pd
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, TypedDict

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

# .env 파일 로드
load_dotenv()


class TopicPageItem(BaseModel):
    """ Classify ['Topic', 'Page'] on documents."""
    check_topics_pages: List[Any] = None
    

def pdf_file_loader(pdf_file_path):

    start = time.time()

    print(pdf_file_path)

    #docs = pymupdf4llm.to_markdown(pdf_file_path, page_chunks=True, show_progress=False)
    docs = pymupdf4llm.to_markdown(pdf_file_path, page_chunks=True, show_progress=True)
    
    sec = time.time() - start
    times = str(datetime.timedelta(seconds=sec)) # 걸린시간 보기좋게 바꾸기
    short = times.split(".")[0] # 초 단위 까지만
    print(f"{times} sec")
    print(f"{short} sec")
    
    return docs


def text_preprocessing(text):
    # pymupdf4llm '�' --> ' ' 로 대체
    text = text.replace('�', ' ')

    # pymupdf4llm '\n\n' --> ' ' 로 대체
    text = text.replace('\n\n', ' ')

    return text


def filtering_page_split_contents(docs):
    
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")

    MAX_PAGE = max(65, int(len(docs) * 0.15))
    if MAX_PAGE > len(docs):
        MAX_PAGE = len(docs)

    # 본문의 내용이 많은 page를 포함하게 되면, max_length에 걸려 5로 설정
    BATCH_SIZE = 5
    
    data_list = []
    ## 약관의 pages 가 많아 50 page 와 전체 page의 15% 중 큰 값까지 범위 설정
    
    for i in range(0, MAX_PAGE):
        
        ## 빈 페이지 또는 Short Text Filtering
        if len(docs[i]['text']) > 10:
            
            # text_preprocessing
            data_list.append(text_preprocessing(docs[i]['text']))
    
    print(f"MAX_PAGE : {MAX_PAGE}")
    print(f"Filtering PAGE : {len(data_list)}")
    

    parser = PydanticOutputParser(pydantic_object=TopicPageItem, encoding='utf-8')

    with open("contents_prompt_template", "r") as f:
        contents_prompt_template = f.read()

    prompt = PromptTemplate(
        template=contents_prompt_template,
        input_variables=["doc"],
        partial_variables={"format": parser.get_format_instructions()}
    )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # chain 을 생성합니다.
    chain = prompt | llm | parser

    contents_list = []

    tpm_count = 0
    
    for i in range((len(data_list)//BATCH_SIZE)+1):

        sum_text = "\n".join(data_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]) 

        token_count = encoder.encode(sum_text)        
        tpm_count += len(token_count)

        print("tpm_count : ", tpm_count)

        ## TPM Limit 걸리는 부분 해결을 위해 TPM 확인
        if tpm_count > 30000:
            time.sleep(30)
            tpm_count = 0

        
        response = chain.batch(
            data_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        )
        print(i, len(response))
        
        
        #for j in range(len(response)):
        for item in response:
            if len(item.check_topics_pages) > 1:
                for content in item.check_topics_pages:
                    print(content)
                    contents_list.append(content)
                #
            #print(i, response[i].check_topics_pages, len(response[i].check_topics_pages))
    
    ## page 낮은 순으로 sort
    if len(contents_list) > 0:
        contents_list.sort(key=lambda x : x[1])
    
    return contents_list


def matching_contents_docs(contents_list, docs):
    ## 현재 page ~ 다음 page 포함
    ## 분류 오차로 인해 page가 1 이라면...??
    ## 2, 3, 4, 5 라면...?? 

    ## contents가 없는것 작성 필요


    
    documents = []
    contents_json = {"sections": []}
    PAGE_COUNT = 1
    INIT_PAGE = contents_list[0][1]
    
    if PAGE_COUNT < INIT_PAGE:
        while PAGE_COUNT < INIT_PAGE:
            """
            documents.append(
                Document(
                    page_content=text_preprocessing(docs[PAGE_COUNT-1]['text']),
                    metadata={
                        'source': docs[PAGE_COUNT-1]['metadata']['file_path'],
                        'topic': 'Remove',
                        'page': docs[PAGE_COUNT-1]['metadata']['page'],
                    }
                )
            )
            """
            PAGE_COUNT += 1
    
    for i in range(len(contents_list)):
        ## page 1부터 start, docs index 는 0부터 start
        ## 목차 맨 처음 page 이전은, 의미 없는 정보라고 가정하여 topic에 Remove 표시

        contents_json["sections"].append({"title": contents_list[i][0],"page": contents_list[i][1]})
    
        PAGE_COUNT = START_PAGE = contents_list[i][1]
    
        if i < len(contents_list) - 1:        
            END_PAGE = contents_list[i+1][1]
        else:
            END_PAGE = len(docs)
    
        while (PAGE_COUNT >= START_PAGE) and (PAGE_COUNT <= END_PAGE):
            print("PAGE_COUNT, END_PAGE : ", PAGE_COUNT, END_PAGE)
            try:
                documents.append(
                    Document(
                        page_content=text_preprocessing(docs[PAGE_COUNT-1]['text']),
                        metadata={
                            'source': docs[PAGE_COUNT-1]['metadata']['file_path'],
                            'insurance_name': docs[PAGE_COUNT-1]['metadata']['file_path'].split('/')[-1][:-4],
                            'topic': contents_list[i][0],
                            'page': docs[PAGE_COUNT-1]['metadata']['page'],
                        }
                    )
                )
                PAGE_COUNT += 1
            except:
                return documents, contents_json

    return documents, contents_json

"""
def trans_contents_list_json(contents_list):

    contents_json = {"sections": []}

    for i in range(len(contents_list)):
        contents_json["sections"].append({"title": contents_list[i][0],"page": contents_list[i][1]})
        
    #contents_json = json.dumps(contents_json)

    return contents_json
"""

def split_doc_to_contents(pdf_path, data_list):

    total_origin_docs = []
    total_documents = []
    total_contents_json_list = []
    for data in data_list:
        code = data["code"]
        file_path = os.path.join(pdf_path, code, data["pdf"])
        
        docs = pdf_file_loader(file_path)
        total_origin_docs.append(docs)
        contents_list = filtering_page_split_contents(docs)
        #contents_json = trans_contents_list_json(contents_list)
        print("contents_list : ")
        print(contents_list)
        
        matching_documents, contents_json = matching_contents_docs(contents_list, docs)
        total_contents_json_list.append(contents_json)
        #print("matching_documents : ")
        
        for i in range(len(matching_documents)):
            total_documents.append(matching_documents[i])
            #print(matching_documents[i])
    
    return total_origin_docs, total_documents, total_contents_json_list


def file_check(vector_db_path, pdf_path, contents_path):

    ## 1. 처음 작업
    ## vector_db_path 에서 'checkpoint_data.pickle' 파일이 있는지 확인
    ## 파일이 없으면 처음 작업이라고 확정함
    ## vector_db_path 에 'checkpoint_data.pickle' 파일 생성

    ## 2. 이어서 작업(작업 후 추가)
    ## 파일이 있으면 이어서 작업
    ## 3. 이어서 작업시, 이전에 작업한 파일들 제외하고 추가함

    # load and uncompress.
    checkpoint_data_file_path = os.path.join(vector_db_path, 'checkpoint_data.pickle')
    if os.path.isfile(checkpoint_data_file_path):
        with gzip.open(checkpoint_data_file_path, 'rb') as f:
            checkpoint_df = pickle.load(f)

    ## pdf file path 에서 폴더 리스트를 뽑고, contents 를 json으로 만들기 위한
    ## json 폴더를 pdf file path와 같은 이름으로 만들어줌
    folder_list = os.listdir(pdf_path)

    data_list = []
    for folder in folder_list:
        
        ## pdf file 의 code 폴더를 json 폴더에도 생성
        os.makedirs(os.path.join(contents_path, folder), exist_ok=True)
        
        files = os.listdir(os.path.join(pdf_path, folder))

        for file_name in files:
            if ".pdf" in file_name:
                data = [folder, file_name, file_name[:-4] + ".json"]
                data_dict = {"code": folder, "pdf": file_name, "json": file_name[:-4] + ".json"}

            
                ## 파일이 있으면 이어서 작업
                if os.path.isfile(checkpoint_data_file_path):
                    ## dataframe에 작업했던 내용이 있는지 확인
                    ## 없으면 작업해야 하니 data_list에 append
                    if checkpoint_df.isin(data).any().all() == False:
                        data_list.append(data_dict)
              
                else:
                    data_list.append(data_dict)

    ## 파일이 있으면 이어서 작업
    if os.path.isfile(checkpoint_data_file_path):
        if len(data_list) > 0:
            checkpoint_df = pd.concat([checkpoint_df, pd.DataFrame([data_list])], ignore_index=True)
            
    ## 파일이 없으면 새로 작업
    else:
        if len(data_list) > 0:
            checkpoint_df = pd.DataFrame(data_list)

        #with gzip.open(checkpoint_data_file_path, 'wb') as f:
        #    pickle.dump(checkpoint_df, f)
        
                
    return data_list, checkpoint_df


def save_contents(contents_path, data_list, contents_list_json):

    for i in range(len(contents_list_json)):
        code = data_list[i]["code"]
        file_path = os.path.join(contents_path, code, data_list[i]["json"])

        with open(file_path, 'w') as f:
            json.dump(contents_list_json[i], f, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    """
    ## 해당 폴더 하위 전부
    file_path = "/workspace/home/jhko/medihub/pdf_data"

    pdf_file_path_list = []
    for (path, dirs, files) in os.walk(file_path):        
        for file_name in files:
            if ".pdf" in file_name:
                pdf_file_path_list.append(os.path.join(path, file_name))
    """
    pdf_file_path_list = ["/workspace/home/jhko/medihub/pdf_data/스마트변액유니버설CI종신보험(무배당)_조화윤.pdf"]

    matching_documents = split_doc_to_contents(pdf_file_path_list)


    #for file_path in file_path_list:
    #    documents = ...
    
    