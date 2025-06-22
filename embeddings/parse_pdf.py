import pdfplumber
import os
import re

def extract_product_id(text):
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        if "Product datasheet" in line:
            for next_line in lines[idx+1:]:
                next_line = next_line.strip()
                if next_line:
                    return next_line
    return None

def extract_technical_sections(text, section_headers=None):
    if section_headers is None:
        section_headers = [
            "Areas of application",
            "Product features and benefits",
            "Technical Data",
            "Photometric Data",
            "Electrical Data",
            "Physical Attributes",
            "Operating Conditions",
            "Lifetime Data",
            "Environmental & Regulatory Information",
            "Safety advice",
            "Logistical Data"
        ]
    pattern = r"(" + "|".join([re.escape(h) for h in section_headers]) + r")"
    results = []
    lines = text.split('\n')
    chunk = []
    current_header = None
    for line in lines:
        line_stripped = line.strip()
        header_match = re.match(pattern, line_stripped, re.IGNORECASE)
        if header_match:
            if current_header and chunk:
                results.append({
                    "header": current_header,
                    "content": "\n".join(chunk).strip()
                })
            current_header = header_match.group(1)
            chunk = [line_stripped]
        elif current_header:
            chunk.append(line_stripped)
    if current_header and chunk:
        results.append({
            "header": current_header,
            "content": "\n".join(chunk).strip()
        })
    return results

def parse_pdf_for_tech_sections(pdf_path):
    source = os.path.basename(pdf_path)
    all_sections = []
    with pdfplumber.open(pdf_path) as pdf:
        first_page_text = pdf.pages[0].extract_text()
        product_id = extract_product_id(first_page_text) if first_page_text else None

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                sections = extract_technical_sections(text)
                for sec in sections:
                    # Add Product ID on each section
                    section_content = f"Product_ID: {product_id}\n{sec['content']}"
                    chunk = {
                        "header": sec['header'],
                        "content": section_content,
                        "page": page_num,
                        "product_id": product_id,
                        "source": source,
                    }
                    all_sections.append(chunk)
    return all_sections

if __name__ == "__main__":
    all_sections = parse_pdf_for_tech_sections('../pdfs/dataset_1/ZMP_1004795.pdf')
    for section in all_sections:
        print(section)
