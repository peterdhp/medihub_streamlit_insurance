# 80% CPU 코어 사용하기
# convert the document to markdown
import os
import time
import random
import pickle
import datetime
from collections import Counter
from multiprocessing import Pool, cpu_count

import parmap
import pymupdf4llm

import fitz
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

def make_trans_path(input_path):

    split_path = input_path.split('/')    
    split_path[2] = 'insurance_text'
    
    trans_path = '/'
    for i in range(1, len(split_path)):
        if (i == len(split_path) - 1):
            trans_path = os.path.join(trans_path, split_path[i][:-4] + '.pickle')
        else:        
            trans_path = os.path.join(trans_path, split_path[i])
            os.makedirs(trans_path, exist_ok=True)

    #print("trans_path : ", trans_path)
    return trans_path


def get_pdf_page_count(pdf_path):
    """
    Efficiently get the total number of pages in a PDF using PyMuPDF.
    """
    with fitz.open(pdf_path) as pdf:
        return pdf.page_count


def extract_num_from_text(text):
    # """
    # Attempt to find a page number in the page text.
    # Example patterns:
    #   - 'Page 12'
    #   - a line that's just '12'
    # This is *very* naive; you might need a more robust regex or logic.
    # """
    # # Look for 'Page X' pattern (case-insensitive)
    # match = re.search(r"(?i)\bpage\s+(\d+)\b", text)
    # if match:
    #     return int(match.group(1))
    
    # # Or look for a line that only contains a single number (again, naive).
    # single_num_match = re.search(r"(?m)^\s*(\d+)\s*$", text)
    # if single_num_match:
    #     return int(single_num_match.group(1))
    
    # return None
    
    class Page(BaseModel):
        pagenumber: int | None = Field(
        #pagenumber: int = Field(
            description="A single integer representing the page number or None if not found."
        )
    
    # Define the LLM with structured output
    llm4omini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    pagenumber_extractor = llm4omini.with_structured_output(Page)
        
    pagenumber_extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an assistant designed to extract a single “page number” from text taken from a PDF page. 
Your job is to look for clues in the text—such as “Page 9”, “p. 9”, “9/20”, or a lone number in a footer/header—that likely indicate the actual printed page number on the physical document.

**Instructions**:
1. You will receive the text of one PDF page.
2. You must determine if there is a clear, single “page number” in the text.
   - A “page number” is typically a small integer (often 1–999) that denotes the page’s label in a document.
   - It might appear in formats like “Page 12”, “12”, “- 12 -”, “p.12”, “12/50”, etc.
3. If you find exactly one likely page number, output only that integer.
4. If no clear page number is found, or if there are multiple conflicting page numbers, output “None”.
5. Do not provide any additional text, explanation, or formatting besides the integer or “None”.

**Examples**:
- If the page text is: “This is page 12 of the company handbook.” then return `12`.
- If the page text is: “Chapter 2. Footer: Page 34. Section …” then return `34`.
- If the text does not mention any numeric page reference or has multiple conflicting references (e.g., “Page 5” in the header and “page 6” in the footer), return `None`.

**Your output**:
- Must be exactly one line: either the integer (e.g., `12`) or `None`.

Follow these instructions precisely. Do not add any extra commentary."""),
        ("user", 
         "{page_text}")
    ])
    pagenumber_extractor = pagenumber_extractor_prompt | pagenumber_extractor
    
    result = pagenumber_extractor.invoke({"page_text": text})
    page = result.pagenumber
    
    return page


def extract_offset(pdf_path, page_indexes):
    """
    Extract text from a single page (page_index is 1-based in this example for clarity).
    Then attempt to find a numeric page label in the text.
    """

    # pdfminer doesn't directly let you get 'just one page' easily,
    # but you can break out of the iteration early if you only want that page.
    offsets = []
    docs = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=False, pages=page_indexes, margins=(0, 0, 0, 0))

    for doc in docs:
        page_text = doc["text"]
        page_num = doc["metadata"]["page"]  
        
        #extract the numeric page from text
        #result, extracted_page_num = extract_num_from_text(page_text)
        try:
            extracted_page_num = extract_num_from_text(page_text)
            
        except:
            extracted_page_num = None

        if extracted_page_num is not None:
            offsets.append(page_num - extracted_page_num)
        
    return offsets

def check_offset_consistency(pdf_path, sample_size=10, random_sample=True):
    """
    1. Determine total page count.
    2. Select a sample of pages (random or regular interval).
    3. For each sampled page, extract a textual page number.
    4. Determine if there's a consistent offset (Physical page - Extracted page).
    5. If consistent, return that offset; else return None (representing 'unavailable').
    """
    
    total_pages = get_pdf_page_count(pdf_path)
    if total_pages == 0:
        return None  # or raise an error
    
    # Decide which pages to sample
    if random_sample:
        # pick random pages
        print("total_pages : ", total_pages)
        # sampled_pages = sorted(random.sample(range(1, total_pages), min(sample_size, total_pages)))
        # sampled_pages = sorted(random.sample(range(1, total_pages), min(sample_size, total_pages - 1 if total_pages <= 10 else total_pages)))
        sampled_pages = sorted(random.sample(range(1, total_pages), min(sample_size, total_pages - 1 if total_pages <= sample_size else total_pages)))
    else:
        # pick evenly spaced pages
        interval = max(total_pages // sample_size, 1)
        # e.g., pages at 1/n, 2/n... etc. This is a simple approach
        # so let's just pick every 'interval' page up to sample_size times
        sampled_pages = [p for p in range(1, total_pages+1, interval)]
        # If that yields more than sample_size because total_pages is big, just take the first sample_size.
        sampled_pages = sampled_pages[:sample_size]
    
        # Extract textual page num from the page's content
    offsets = extract_offset(pdf_path, sampled_pages)
    
    # Check if all offsets are the same
    #if all(o == offsets[0] for o in offsets):
    #    return offsets[0]  # consistent offset
    offset_counts = Counter(offsets)
    #print("offset_counts : ", offset_counts)
    #print("len(offset_counts) : ", len(offset_counts))

    if len(offset_counts) > 0:
        ## 최대 빈도수의 값 선택, but 같은 count 일 경우... 제일 작은값 or index 순서로 선택...
        most_common_offset, count = offset_counts.most_common(1)[0]

        if count >= min(3, (total_pages-1)//3):
            return most_common_offset
        else:
            return None  # not consistent
    else:
        return None


def pdf2text(data_chunk):

    for dc in data_chunk:
        try:  # 파일 처리 중 발생할 수 있는 예외 처리

            docs = pymupdf4llm.to_markdown(dc, page_chunks=True, show_progress=False)

            offset = check_offset_consistency(dc, sample_size=20, random_sample=True)
            print("offset = ", offset)
            for doc in docs:
                doc['metadata']['offset'] = offset
            
            pickle_file_path = make_trans_path(dc)
            #print("pickle_file_path : ", pickle_file_path)
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(docs, f, pickle.HIGHEST_PROTOCOL)

        except Exception as e:  # 모든 예외를 잡아서 처리
            print(f"Error processing file {dc}: {e}")  # 에러 메시지 출력
            continue  # 다음 파일로 넘어감
        except (fitz.fitz.FileNotFoundError, pymupdf4llm.exceptions.FileNotFound): #파일이 없을경우 에러처리
            print(f"Error processing file {dc}: File Not Found")
            continue

    
    #return result

if __name__ == "__main__":
    
    data_path = []
    folder_path = '/workspace/insurance'
    print("folder_path : ", folder_path)
    for (path, dirs, files) in os.walk(folder_path):
        print("files_len : ", len(files))
        for file_name in files:
            if '.pdf' in file_name:
                t = make_trans_path(os.path.join(path, file_name))
                if os.path.isfile(t):
                    pass
                else:
                    data_path.append(os.path.join(path, file_name))

    data_path.sort()
    print("data_path : ", len(data_path))
    num_cores = int(cpu_count() * 0.5)  # 시스템의 CPU 코어 수 확인
    print("num_cores : ", num_cores)
    data_path_chunks = [data_path[i::num_cores] for i in range(num_cores)]  # 데이터를 CPU 코어 수만큼 청크로 나눔

    results = parmap.map(pdf2text, data_path_chunks, pm_pbar=True, pm_processes=num_cores)
