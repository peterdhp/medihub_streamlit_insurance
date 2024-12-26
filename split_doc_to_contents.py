import os
import time
import datetime
import pymupdf4llm
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, TypedDict
from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


from typing import List, Dict, Any, Optional
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import json

# .env 파일 로드
load_dotenv()


class TopicPageItem(BaseModel):
    """ Classify ['Topic', 'Page'] on documents."""
    check_topics_pages: List[Any] = None
    

def pdf_file_loader(pdf_file_path):

    start = time.time()

    print(pdf_file_path)

    docs = pymupdf4llm.to_markdown(pdf_file_path, page_chunks=True, show_progress=False)
    
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
    MAX_PAGE = max(50, int(len(docs) * 0.15))

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

    for i in range((len(data_list)//BATCH_SIZE)+1):
        
        response = chain.batch(
            data_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        )
        print(i, len(response))
        
        #for j in range(len(response)):
        for item in response:
            if len(item.check_topics_pages) > 1:
                for content in item.check_topics_pages:
                    contents_list.append(content)
                #
            #print(i, response[i].check_topics_pages, len(response[i].check_topics_pages))
    
    ## page 낮은 순으로 sort
    contents_list.sort(key=lambda x : x[1])
    
    return contents_list


def matching_contents_docs(contents_list, docs):
    ## 현재 page ~ 다음 page 포함
    ## 분류 오차로 인해 page가 1 이라면...??
    ## 2, 3, 4, 5 라면...?? 
    
    documents = []
    PAGE_COUNT = 1
    INIT_PAGE = contents_list[0][1]
    
    if PAGE_COUNT < INIT_PAGE:
        while PAGE_COUNT < INIT_PAGE:
            documents.append(
                Document(
                    page_content=text_preprocessing(docs[PAGE_COUNT-1]['text']),
                    metadata={
                        'source': os.path.basename(docs[PAGE_COUNT-1]['metadata']['file_path']),
                        'topic': 'Remove',
                        'page': docs[PAGE_COUNT-1]['metadata']['page'],
                    }
                )
            )
            PAGE_COUNT += 1
    
    for i in range(len(contents_list)):
        ## page 1부터 start, docs index 는 0부터 start
        ## 목차 맨 처음 page 이전은, 의미 없는 정보라고 가정하여 topic에 Remove 표시
    
        PAGE_COUNT = START_PAGE = contents_list[i][1]
    
        if i < len(contents_list) - 1:        
            END_PAGE = contents_list[i+1][1]
        else:
            END_PAGE = len(docs)
    
        while (PAGE_COUNT >= START_PAGE) and (PAGE_COUNT <= END_PAGE):
            documents.append(
                Document(
                    page_content=text_preprocessing(docs[PAGE_COUNT-1]['text']),
                    metadata={
                        'source': os.path.basename(docs[PAGE_COUNT-1]['metadata']['file_path']),
                        'topic': contents_list[i][0],
                        'page': docs[PAGE_COUNT-1]['metadata']['page'],
                    }
                )
            )
            PAGE_COUNT += 1

    return documents

def split_doc_to_contents(pdf_file_path_list):

    total_documents = []
    
    for file_path in pdf_file_path_list:
        docs = pdf_file_loader(file_path)
        contents_list = filtering_page_split_contents(docs)
        matching_documents = matching_contents_docs(contents_list, docs)
        print("matching_documents : ")
        #for i in range(len(matching_documents)):
            #print(matching_documents[i])
    
    return matching_documents
        


def save_documents(documents, file_path):
    """
    Save the documents to a JSON file for later retrieval.

    :param documents: List of documents with metadata.
    :param file_path: Path to save the JSON file.
    """
    serialized_docs = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serialized_docs, f, ensure_ascii=False, indent=4)

def load_documents(file_path):
    """
    Load the documents from a JSON file.

    :param file_path: Path to the JSON file.
    :return: List of loaded documents.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    pdf_file_path_list = ["무배당 메리츠 듬뿍담은 순환계질환보장보험2112(2종)약관.pdf"]
    matching_documents = split_doc_to_contents(pdf_file_path_list)
    
    # Save the matching documents
    save_path = "documents/example_doc.json"
    save_documents(matching_documents, save_path)
    print(f"Documents saved to {save_path}")

    # Later, load the documents
    loaded_documents = load_documents(save_path)
    print(f"Loaded {len(loaded_documents)} documents.")