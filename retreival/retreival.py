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
import re
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from prompts.rephraser import REPHRASE_QUERY_PROMPT

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)


def retreive(user_input: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "electric-products-sections-productid"

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    llm = ChatOpenAI(
        model_name="gpt-4.1-mini",
        temperature=0,
        max_retries=2,

    )

    llm_runnable = REPHRASE_QUERY_PROMPT | llm

    structured_out = llm_runnable.invoke({"input": user_input})

    # Clean JSON
    content = structured_out.content
    clean_content = re.sub(r"^```json|^```|```$", "", content, flags=re.MULTILINE).strip()

    print("INFO:: Structured Output:", structured_out)
    result = json.loads(clean_content)
    rephrased_query = result["rephrased_query"]
    product_id = result["product_id"]
    header = result["header"]

    print("INFO:: Rephrased:", rephrased_query)
    print("INFO:: Product ID:", product_id)
    print("INFO:: Header:", header)

    # Dynamic FIltering
    filter_kwargs = {}
    #if product_id:
    #  filter_kwargs["product_id"] = product_id
    if header:
        filter_kwargs["header"] = header

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 20,
            "filter": filter_kwargs if filter_kwargs else None
        }
    )

    compressor = RankLLMRerank(top_n=5, model="gpt", gpt_model="gpt-4.1-mini")
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

    response = retrieval_chain.invoke({"input": rephrased_query})
    return response


if __name__ == "__main__":
    retreive("What is the color Temperature of SIRIUS HRI 330W 2/CS 1/SKU?")
