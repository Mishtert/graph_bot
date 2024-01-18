import glob
import os
import re

from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from utils.load_model import embeddings, model
from utils.prompt_config import enterprise_mq_query_prompt

data_path = os.path.join(os.getcwd(), "data", "raw_data", "emp_policy")


def read_create_docs(directory=data_path):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a very small chunk size, just to show.
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    documents = []
    for file in pdf_files:
        policy_name = os.path.basename(file)
        loader = PyPDFLoader(file)
        docs = loader.load_and_split(text_splitter)
        for doc in docs:
            doc.metadata['source'] = f"{policy_name}"
        documents.extend(docs)
    return documents


# await read_create_docs()
#
# # Get the current working directory
# base_dir = os.getcwd()
# # Construct the path dynamically
# config_path = os.path.join(base_dir, "multi_skill_bot", "config.env")
# enterprise_documents = read_create_docs()
#

def mq_string(question, mq_query_prompt=enterprise_mq_query_prompt):
    # Load custom parser
    output_parser = LineListOutputParser()
    # Create multi queries
    mqs = LLMChain(llm=model, prompt=mq_query_prompt,
                   output_parser=output_parser).run(question).lines
    # Add original question to multi queries
    mqs.insert(0, question)
    # Regular expression to match serial numbers (e.g., '1.', '2.', etc.)
    pattern = r'^\d+\.\s'
    # Apply the function to each string in the list using a list comprehension
    cleaned_mqs = [re.sub(pattern, '', s) if re.match(pattern, s) else s for s in mqs]
    q_string = '\n'.join(cleaned_mqs)
    # print(q_string)
    return q_string


def get_retriever(index_path):
    db = FAISS.load_local(index_path, embeddings)
    # db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_type="mmr",
                                search_kwargs={"fetch_k": 20, "k": 5})
    _filter = LLMChainFilter.from_llm(model)
    compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)
    return compression_retriever


def reorder(docs):
    reordered_docs = LongContextReorder().transform_documents(docs)
    return reordered_docs


def initialized_agent(toolkit, agent_kwargs):
    sql_agent = initialize_agent(
        toolkit.get_tools(),
        model,
        verbose=True,
        agent_kwargs=agent_kwargs,
        handle_parsing_errors=True
    )
    return sql_agent
