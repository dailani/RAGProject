import pdfplumber
import os

def parse_pdf(pdf_path):
    source = os.path.basename(pdf_path)
    all_rows = []
    all_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"\n=== Processing Page {page_num} ===")

            # FIRST PAGE: always extract text
            if page_num == 1:
                text = page.extract_text()
                if text:
                    print(f"\n--- Main Text on Page {page_num} ---\n{text.strip()}")
                    all_texts.append({
                        "page": page_num,
                        "text": text.strip(),
                        "source": source
                    })

            else:
                text = page.extract_text() or ""
                # Check if 'safety advice' is in the page text
                if "safety advice" in text.lower():
                    print(f"\n--- Safety Advice Text on Page {page_num} ---\n{text.strip()}")
                    all_texts.append({
                        "page": page_num,
                        "text": text.strip(),
                        "source": source
                    })
                else:
                    # Only extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables, start=1):
                            print(f"\n--- Table {table_idx} on Page {page_num} ---")
                            for row in table:
                                print(row)
                            if table and len(table) > 1:
                                headers = table[0]
                                for row in table[1:]:
                                    row_dict = dict(zip(headers, row))
                                    row_dict["page"] = page_num
                                    row_dict["source"] = source
                                    all_rows.append(row_dict)
    return all_rows, all_texts




