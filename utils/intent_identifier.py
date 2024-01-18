from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool

from utils.base_utils import mq_string
from utils.load_model import model


@tool
def identify_intent(query: str) -> str:
    """
    Identify the intent of the user based on the query and recommend next steps
    """
    intent_prompt = """You are an expert language analyst who can understand the intent of the question. 
                        If you do not understand the question, ask the user to rephrase. Do not assume.
                        
                        Check if the question is a followup to the previous question. If it is not a followup,
                        identify the intent. Do not run on all chains by default.
                        You have access to below data:
                        -----
                        health_insurance_db.db with table name health_insurance_data
                        employee_leaves.db with table name employee_leaves_data
                        IT_support.db with table name IT_support_tickets_data
                        employee policy details
                        -----
                        Given the below question:
                        -----
                        {question}
                        -----
                        Please identify the intent based on the classifications below:
                        -----
                         - small talk
                         - policy related
                         - leaves related
                         - insurance related
                         - IT support related
                         - information available in policy documents, but requires look up in leave database first
                         - information available in policy documents, but requires look up in IT support ticket database first
                         - information available in policy documents, but requires look up in health insurance database first
                         - Looking for information about some other User ID
                         - None of the above
                        -----
                        and recommend the next steps based on the key:value pair you see below:
                        -----
                        small talk: perform small talk. 
                        None of the above : Do not recommend any other chain.End of Chain. Wait for next question. Ask user to rephrase the question.
                        Looking for information about some other User ID: ``Respond saying I can't answer that question.Ask for informaiton only about yourself``
                        policy related : search company policies
                        leaves related : search leaves database
                        insurance related : search health insurance database
                        IT support related : search IT support ticket database
                        information available in policy documents, but requires look up in leave database first : search leave database and then use that information to search the policy documents
                        information available in policy documents, but requires look up in IT support ticket database first : search IT support ticket database and then use that information to search the policy documents
                        information available in policy documents, but requires look up in health insurance database first : search health insurance database and then use that information to search the policy documents
                        Note: 
                         - If there is no information available, say `I don't know`.
                         - Answer questions only related to greetings, policies, leaves, health insurance, IT support tickets.
                         - Recommendation is for None of the above, you can engage in greetings related talk and inform the user how you can support him/her.
                        """
    custom_prompt = PromptTemplate(
        template=intent_prompt, input_variables=["question"]
    )
    q_string = mq_string(query)
    llm_chain = LLMChain(llm=model, prompt=custom_prompt )
    result = llm_chain({"question":q_string})
    print(result['text'])
    return result['text']

# identify_intent("Hi my name is Bob")