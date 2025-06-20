import getpass
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY_API_KEY"))



if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def retrieve():
    index_name = "electric-products-index"
    index = pc.Index(index_name)
    print(index.describe_index_stats())

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #Need translation
    query = "What is the color temperature of SIRIUS HRI 330W 2/CS 1/SKU?"

    query_embedding = embeddings.embed_query(query)

    print(query_embedding)

    results = index.query(
    namespace="__default__",
    vector=query_embedding,
    top_k=2,
    include_metadata=True,
    include_values=False
)

    print(results)
    return  results


if __name__ == "__main__":
    retrieve()