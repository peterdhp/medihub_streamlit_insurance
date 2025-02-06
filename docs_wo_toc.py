import re
def text_preprocessing(text):
    # pymupdf4llm '�' --> ' ' 로 대체
    text = text.replace('�', ' ')
    
    # Add extra line before and after tables
    text = re.sub(r'(\n\s*\|)', r'\n\1', text)
    text = re.sub(r'(\|\s*\n)', r'\1\n', text)

    # Convert double line spacing to single line spacing
    text = re.sub(r'\n{2,}', '\n', text)

    cleaned_text = re.sub(r'[^\S\n]+', ' ', text)

    # Remove lines of only dashes (like "-----").
    cleaned_text = re.sub(r'^\s*-{2,}\s*$\n?', '', cleaned_text, flags=re.MULTILINE)

    # Regex: Remove entire lines that are purely (optional heading) + (page number).
    page_number_pattern = re.compile(
        r'''(?ix)               # Case-insensitive (i), verbose (x)
        ^
        \s*
        (?:\#{1,6}\s+)?          # Optional Markdown heading(s) (#, ##, ..., up to 6) + space
        (?:Page|p\.?|pg\.?|vol\.?|section)?  # Optional prefix word
        \s*
        [()\[\]{}*\~\-\—]*      # Optional punctuation (simplified)
        \s*
        (?:\d+|[IVXLCDM]+)      # Digits OR Roman numerals
        \s*
        [()\[\]{}*\~\-\—]*      # Optional closing punctuation (simplified)
        \s*
        $                       # End of line
        ''',
        re.MULTILINE  # Flags are passed directly here
    )
    
    cleaned_text = page_number_pattern.sub('', cleaned_text)

    # Clean up extra blank lines
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text).strip()


    return cleaned_text