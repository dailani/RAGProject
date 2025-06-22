import os
import getpass
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from embeddings.parse_pdf import parse_pdf_for_tech_sections
from utils.pdf_path import get_all_pdfs_path

load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def flatten_sections_to_documents(section_chunks_param):
    flatten_docs = []
    for section in section_chunks_param:
        page_content = section['content']
        flatten_docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "page": section.get("page"),
                    "product_id": section.get("product_id"),
                    "header": section.get("header"),
                    "source": section.get("source")
                }
            )
        )
    return flatten_docs


def embbeding_pipeline():
  index_name = "electric-products-sections-productid"

  if not pc.has_index(index_name):
    pc.create_index(
      index_name,
      metric="cosine",
      spec=ServerlessSpec(cloud="aws", region="us-east-1"),
      dimension=3072
    )

  docs = []
  all_pdf_paths = get_all_pdfs_path()
  for pdf_path in all_pdf_paths:
    section_chunks = parse_pdf_for_tech_sections(pdf_path)
    docs.extend(flatten_sections_to_documents(section_chunks))

  embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
  batch_size = 100
  for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    PineconeVectorStore.from_documents(
      batch,
      embeddings,
      index_name=index_name,
    )
  print(f"Uploaded {len(docs)} documents in batches of {batch_size} to index '{index_name}'.")


if __name__ == "__main__":
  embbeding_pipeline()