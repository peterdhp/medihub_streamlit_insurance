import random
import fitz
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pymupdf4llm
from operator import itemgetter
from langchain_core.runnables import RunnableParallel
from collections import Counter

load_dotenv()

def extract_num_from_text(texts):
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
    pages =[]
    
    class Page(BaseModel):
        pagenumber: int | None = Field(
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
    
    tasks = {}
    for i, text in enumerate(texts):
        task_name = f'task{i}'
        tasks[task_name] = ({'page_text': itemgetter(f'text{i}')}) | pagenumber_extractor

    parallel_runner = RunnableParallel(**tasks)
    input_dict = {}
    for i in range(len(texts)): 
        input_dict[f'text{i}'] =  texts[i]
    
    results = parallel_runner.invoke(input_dict)
    for result in results.values():
        if result is not None:
            pages.append(result.pagenumber)
    
    return pages
    
    
def get_pdf_page_count(pdf_path):
    """
    Efficiently get the total number of pages in a PDF using PyMuPDF.
    """
    with fitz.open(pdf_path) as pdf:
        return pdf.page_count

def extract_offset(pdf_path, page_indexes):
    """
    Extract text from a single page (page_index is 1-based in this example for clarity).
    Then attempt to find a numeric page label in the text.
    """
    # pdfminer doesn't directly let you get 'just one page' easily,
    # but you can break out of the iteration early if you only want that page.
    offsets = []
    docs = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=False,pages=page_indexes,margins=(0, 0, 0, 0))
    page_text = []
    page_num = []
    for doc in docs:
        page_text_ = doc["text"]
        page_text.append(page_text_)
        page_num_ = doc["metadata"]["page"]  
        page_num.append(page_num_)
        
        #extract the numeric page from text
    extracted_page_num = extract_num_from_text(page_text)
    
    
    for i in range(len(extracted_page_num)) : 
        if extracted_page_num[i] is not None : 
            offsets.append(page_num[i] - extracted_page_num[i])
            
        
    return offsets, page_num, extracted_page_num

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
        sampled_pages = sorted(random.sample(range(1, total_pages), min(sample_size, total_pages)))
    else:
        # pick evenly spaced pages
        interval = max(total_pages // sample_size, 1)
        # e.g., pages at 1/n, 2/n... etc. This is a simple approach
        # so let's just pick every 'interval' page up to sample_size times
        sampled_pages = [p for p in range(1, total_pages+1, interval)]
        # If that yields more than sample_size because total_pages is big, just take the first sample_size.
        sampled_pages = sampled_pages[:sample_size]
    
        # Extract textual page num from the page's content
    offsets, pdf_page, extracted_page = extract_offset(pdf_path, sampled_pages)
    
    offset_counts = Counter(offsets)
    most_common_offset, count = offset_counts.most_common(1)[0]
    
    if count >= 3:
        return most_common_offset, pdf_page, extracted_page  # most frequent offset
    else:
        return None, pdf_page, extracted_page  # not frequent enough


def main():
    pdf_file = "your_document.pdf" ### replace with your pdf file
    
    offset, pdf_page, extracted_page = check_offset_consistency(pdf_file, sample_size=10, random_sample=True)
    
    if offset is not None:
        print(f"Calculated offset: {offset}")
    else:
        print("unavailable")

if __name__ == "__main__":
    main()