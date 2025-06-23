import os
import getpass
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
import json
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from prompts.rephraser import MULTI_HEADER_PROMPT, VECTOR_REPHRASER_QUERY_PROMPT
import re
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

def extract_product_code_from_query(query):
    match = re.search(r'\b(\d{13})\b', query)
    if match:
        return match.group(1)
    return None


def build_retriever_query(headers_array, user_query):
    product_code = extract_product_code_from_query(user_query)
    print("Extracted Product Code:", product_code)

    if len(headers_array) == 1:
        header_filter = headers_array[0]
        print("Header filter (single):", header_filter)
    else:
        header_filter = {"$or": headers_array}
        print("Header filter (OR):", header_filter)

    # Decides if filter by product_code or  by header
    if product_code:
        filter_kwargs = {"product_code": {"$in": [product_code]}}
        print("Final filter (only product_code):", filter_kwargs)
    else:
        filter_kwargs = header_filter
        print("Final filter (only header):", filter_kwargs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "electric-products-sections-productcode"
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 50,
            "filter": filter_kwargs
        }
    )
    return retriever

def retreive(user_input: str):

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI(
        model_name="gpt-4.1-mini",
        temperature=0,
        max_retries=2,

    )

    llm_runnable_header = MULTI_HEADER_PROMPT | llm

    headers_output = llm_runnable_header.invoke({"input": user_input})
    # Clean JSON
    headers_content = headers_output.content
    headers_array = json.loads(headers_content)

    # Deliver retriver we built previosly
    retriever = build_retriever_query(headers_array, user_input)

    llm_runnable_rephraser = VECTOR_REPHRASER_QUERY_PROMPT | llm
    rephraser_output = llm_runnable_rephraser.invoke({"input": user_input})
    print("Rephraser output:", rephraser_output)

    compressor = RankLLMRerank(top_n=20, model="gpt", gpt_model="gpt-4.1-mini")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=compression_retriever,
        combine_docs_chain=combine_docs_chain
    )

    response = retrieval_chain.invoke({"input": rephraser_output.content})
    print("Final response:", response)
    return response


if __name__ == "__main__":
    retreive("Gebe mir alle Leuchtmittel mit mindestens 1000 Watt und Lebensdauer von mehr als 400 Stunden")
