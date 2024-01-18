from langchain.prompts import PromptTemplate

enterprise_mq_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
       different versions of the given user question to retrieve relevant documents from a vector 
       database. By generating multiple perspectives on the user question, your goal is to help
       the user overcome some of the limitations of the distance-based similarity search. 
       Provide these alternative questions separated by newlines.
       Before generating different versions of questions ensure that these questions are related to queries 
       about information to be found in the enterprise policies. 
       If not do not generate different versions. just pass the original query
       Original question: {question}""",
)

# Run custom stuff chain enterprise_rag_chain (erc)
erc_document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
erc_document_variable_name = "context"

erc_stuff_prompt_override = """

Role: Experienced company representative, proficient at finding information pertinent to any employee queries in your employee policies.
Task: Your task is to respond to the User Query with detailed information using the ``context``  and ``metadata`` provided below.
Tone: Respond to the user query in a professional, courteous, and helpful tone
Adhere strictly to the company's policies and guidelines, and ensure your response is informative, accurate, and formatted according to the provided template.
-----
Context: {context}
-----
Template for Response:
Answer: [Provide a clear and direct answer to the user's inquiry.]
Explanation: [Provide a detailed explanation of the company's policy or guideline, using plain language and avoiding jargon.]
Source: [Reference the source of the information, including document names and page numbers.]

----------------
Given the Context and Instructions above, answer the below User Query with as much detail as possible.
-----
User Query: 
'{query}'
-----
"""

erc_prompt = PromptTemplate(
    template=erc_stuff_prompt_override, input_variables=["context", "query"]
)

