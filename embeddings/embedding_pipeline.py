import os
import getpass
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from embeddings.parse_pdf import parse_pdf
from dotenv import load_dotenv
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from utils.pdf_path import get_all_pdfs_path

load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY_API_KEY"))


def flatten_to_documents(all_rows, all_texts):
  docs = []

  # Process table rows
  for row in all_rows:
    # Clean up (skip empty keys/values, but keep "0"/False if needed)
    row_clean = {k: v for k, v in row.items() if k and v and k not in ["page", "source"]}
    text = "\n".join(f"{k}: {v}" for k, v in row_clean.items())
    docs.append(
      Document(
        page_content=text,
        metadata={
          "page": row.get("page"),
          "source": row.get("source"),
          # You can add any other fields here
        }
      )
    )

  # Process non-table texts
  for t in all_texts:
    text = t["text"]
    docs.append(
      Document(
        page_content=text,
        metadata={
          "page": t.get("page"),
          "source": t.get("source"),
        }
      )
    )

  return docs




if __name__ == "__main__":
  #Step1
  pdf_path = "../pdfs/dataset_1/ZMP_1004795.pdf"
  index_name = "electric-products-index"

  if not pc.has_index(index_name):
    pc.create_index(index_name, metric="cosine" ,spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    dimension=3072)

  docs = []
  all_pdf_paths = get_all_pdfs_path()
  for pdf_path in all_pdf_paths:
    all_rows, all_text = parse_pdf(pdf_path)
    docs.extend(flatten_to_documents(all_rows, all_text))


  #We dont need to text splitters to chunks , because we already transformed each row into a document
  # Step2
  embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

  # Batch upload
  batch_size = 100
  for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    PineconeVectorStore.from_documents(
      batch,
      embeddings,
      index_name=index_name,
    )
  print(f"Uploaded {len(docs)} documents in batches of {batch_size} to index '{index_name}'.")




