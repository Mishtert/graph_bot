import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure
from urllib.parse import quote_plus
from bson import json_util
import json

from utils.load_model import model

_ = load_dotenv('utils/config.env')
default_emp_id = os.getenv("DEFAULT_USER_ID")
employee_id = {
    "Emp_ID": int(default_emp_id)
}
# print(employee_id)

username = quote_plus("davyjdemo")
password = quote_plus("chatdb123!")
uri = f"mongodb+srv://{username}:{password}@cluster0.ddlyqbd.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)



def is_mongodb_running(client):
    try:
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        return True
    except ConnectionFailure:
        return False


print(f"Checking if Mongodb is running: {is_mongodb_running(client)}")


def get_health_info():
    health_db = client["health_insurance"]
    health_collection = health_db['health_insurance_data']
    health_db_info = health_collection.find(employee_id)
    health_info = json.dumps(list(health_db_info), default=json_util.default)
    return health_info


def get_leave_info():
    emp_leave_db = client["employee_leaves"]
    emp_leave_collection = emp_leave_db['employee_leaves_data']
    leave_db_info = list(emp_leave_collection.find(employee_id))
    leave_info = json.dumps(list(leave_db_info), default=json_util.default)
    return leave_info


def get_it_support_info():
    it_db = client["it_support"]
    it_collection = it_db['it_support_ticket_data']
    support_ticket_db_info = list(it_collection.find(employee_id))
    support_ticket_info = json.dumps(list(support_ticket_db_info), default=json_util.default)
    return support_ticket_info


def get_chain():
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
        Context: {context}
        Question: {question}
        """
    )
    chain = (
            RunnablePassthrough.assign(context=(lambda x: x["context"]))
            | prompt
            | model
            | StrOutputParser()
    )
    return chain


@tool
def search_insurance_db(query: str) -> str:
    """
    A function that uses the information from the insurance database and provide answers based on a given query and context.

    Returns:
    str: The result of the model based on the query and the context provided.
    """
    context = get_health_info()
    # print(f"Printing health insurance db extract as context, {context}")
    chain = get_chain()
    result = chain.invoke({
        "question": {query},
        "context": {context}
    })
    return result.strip()


@tool
def search_leave_db(query: str) -> str:
    """
    A function that uses the information from the employee leave database and provide answers based on a given query and context.

    Returns:
    str: The result of the model based on the query and the context provided.
    """
    context = get_leave_info()
    chain = get_chain()
    result = chain.invoke({
        "question": {query},
        "context": {context}
    })
    return result.strip()


@tool
def search_it_support_db(query: str) -> str:
    """
    A function that uses the information from the employee leave database and provide answers based on a given query and context.

    Returns:
    str: The result of the model based on the query and the context provided.
    """
    context = get_it_support_info()
    chain = get_chain()
    result = chain.invoke({
        "question": {query},
        "context": {context}
    })
    return result.strip()
