from langchain.chains import LLMChain, StuffDocumentsChain
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from utils.base_utils import get_retriever, mq_string, reorder
from utils.load_model import model
from utils.prompt_config import erc_document_prompt, erc_document_variable_name, erc_prompt

index_path = "./data/erc_docs"


class SearchEnterprisePolicy(BaseModel):
    query: str = Field(..., description="user query about enterprise policy information from the database")


# (args_schema=SearchEnterprisePolicy)
@tool
def find_policy_answers(query: str) -> str:
    """
    If the question is about generic employee policy this function must be used.
    If there is any employee-specific query, corresponding database information must be considered before looking up the general policy documents here.
    """
    q_string = mq_string(query)
    db_retriever = get_retriever(index_path)
    reordered_docs = reorder(db_retriever.get_relevant_documents(query=query, verbose=False))
    print(reordered_docs)
    # Instantiate the chain
    llm_chain = LLMChain(llm=model, prompt=erc_prompt, )
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=erc_document_prompt,
        document_variable_name=erc_document_variable_name,

    )
    result = chain.run(input_documents=reordered_docs, query=q_string)
    return result

# find_policy_answers("do I have dental benefits?")
