import os

def get_all_pdfs_path():
    pdf_dir = "../pdfs/dataset_1"
    all_paths = []
    for root, dirs, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                all_paths.append(os.path.join(root, file))
    return all_paths

if __name__ == "__main__":
    print(get_all_pdfs_path())
