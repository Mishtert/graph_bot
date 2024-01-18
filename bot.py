import os

from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

from utils.emp_tools import search_insurance_db, search_it_support_db, search_leave_db
from utils.find_policy_answers import find_policy_answers
# from utils.it_support_db_extract import search_it_support_db
# from utils.leave_db_extract import search_leave_db
from utils.load_model import model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish
from langchain.agents import create_openai_functions_agent

from utils.intent_identifier import identify_intent
from utils.small_talk import small_talk

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__3d519106ba654d93b6c25a016a20d827"
os.environ["LANGCHAIN_PROJECT"] = "graph_bot"  # if not specified, defaults to default

# Choose the LLM that will drive the agent
llm = model

# Prompt
prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt)

# Define Tools
tools = [
    small_talk,
    identify_intent,
    find_policy_answers,
    search_it_support_db,
    search_leave_db,
    search_insurance_db
]

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt, )

# Define the agent
agent = RunnablePassthrough.assign(
    agent_outcome=agent_runnable
)


# Define the function to execute tools
def execute_tools(data):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data.pop('agent_outcome')
    # Get the tool to use
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    # Call that tool on the input
    observation = tool_to_use.invoke(agent_action.tool_input)
    # We now add in the action and the observation to the `intermediate_steps` list
    # This is the list of all previous actions taken and their output
    data['intermediate_steps'].append((agent_action, observation))
    return data


# Define logic that will be used to determine which conditional edge to go down

def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"


# Define the graph
from langgraph.graph import END, Graph


def instantiate_chain():
    workflow = Graph()

    # Add the agent node, we give it name `agent` which we will use later
    workflow.add_node("agent", agent)
    # Add the tools node, we give it name `tools` which we will use later
    workflow.add_node("tools", execute_tools)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "exit": END
        }
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge('tools', 'agent')

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    chain = workflow.compile()
    return chain


chain = instantiate_chain()

#
# # Define the chat history
# chat_history = [HumanMessage(content=''), AIMessage(content='')]
#
# result = chain.invoke({
#     "input": "How many leaves do I have?",
#     "intermediate_steps": [],
#     "chat_history": chat_history,
# })
#
# print(result["agent_outcome"].return_values["output"])
#
# new_human_msg = HumanMessage(content=result["input"])
# new_ai_msg = AIMessage(content=output["agent_outcome"].return_values["output"])
# chat_history.append(new_human_msg)
# chat_history.append(new_ai_msg)

# print(output)
