import asyncio
import json
import os

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from bot import chain, instantiate_chain
from utils.example_data import health_insurance_example_data, it_support_example_data, leave_example_data, INFORMATION

_ = load_dotenv('utils/config.env')

# print(f"current working directory: {os.getcwd()}")

# os.environ['DEFAULT_USER_ID'] = '31836'

# print(f"Printing Default UserID from main app: {os.getenv('DEFAULT_USER_ID')}")

with open('./data/emp_details.json', 'r') as f:
    emp_details = json.load(f)
    # print(f"printing employee details, {emp_details}")


# Add like buttons to output
def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


# Set up Gradio Theme
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="red",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """# AI Assisted Conversational Chat For Employees
         1. Get Insights about company policies from **Policy Documents**\n
         2. Get Insights about Health Insurance from **Health Insurance Database**\n
         3. Get Insights about available leaves and availed leaves from **Employee Leave Database**\n
        """)
    with gr.Column(scale=1, variant="panel", elem_id="right-panel"):
        with gr.Tabs() as tabs:
            with gr.TabItem("Employee ID", elem_id="tab-empid", id=0):
                examples_hidden = gr.Textbox(visible=False)
                first_key = list(INFORMATION.values())[0]
                # print(first_key)
                dropdown_samples = gr.Dropdown(first_key,
                                               label="Logged in as:",
                                               value=38433,
                                               interactive=False,
                                               show_label=True,
                                               elem_id="dropdown-samples",
                                               )
                # print(f"{dropdown_samples.value}")
                os.environ['DEFAULT_USER_ID'] = str(dropdown_samples.value)
                print(f"Default User ID From Environment: {os.getenv('default_user_id')}")

    with gr.Tab("Chat Window"):
        with gr.Row(elem_id="chatbot-row"):
            with gr.Column(scale=2):
                # state = gr.State([system_template])
                emp_id = os.getenv('default_user_id')
                emp_name = emp_details[emp_id]
                init_prompt = f"Hello **{emp_name}**, \
                    I am a conversational assistant designed to help you find information. \
                    I can answer your questions about **Company Policies, Leaves, \
                    Health Insurance & IT Support Tickets**."

                chatbot = gr.Chatbot(
                    value=[(None, init_prompt)],
                    show_copy_button=True, show_label=False, elem_id="chatbot",
                    avatar_images=(None, (os.path.join(os.path.abspath(''), "avatar.png"))))

                with gr.Row(elem_id="input-message"):
                    textbox = gr.Textbox(placeholder="Ask me anything here!", show_label=False, scale=1, lines=1,
                                         interactive=True)
                clear = gr.ClearButton([textbox, chatbot])

    # ---------------------------------------------------------------------------------------
    # OTHER TABS
    # ---------------------------------------------------------------------------------------
    with gr.Tab("Leave Database", elem_classes="max-height other-tabs"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(leave_example_data)

    with gr.TabItem("IT Support Ticket", elem_classes="max-height other-tabs"):
        gr.Markdown(it_support_example_data)

    with gr.TabItem("Health Insurance Database", elem_classes="max-height other-tabs"):
        gr.Markdown(health_insurance_example_data)

    mandatory_query = (f"Think step-by-step before deciding proceeding. Does this query requires any employee specific "
                       f"information before looking at general policies, if yes get that information and correlate if "
                       f"necessary with general policies before you answer the following question. \n\n")


    def respond(message, chat_history, mq=mandatory_query):
        # Define the chat history
        _chathistory = [HumanMessage(content=''), AIMessage(content='')]
        # print(f"Chat History: {_chathistory}")  # Debug (initializing chat history)
        question = f"{mq}" + f"{message}"
        # print(f"Question: {question}")
        bot_message = chain.invoke({
            "input": {question},
            "intermediate_steps": [],
            "chat_history": _chathistory
        })
        result = bot_message["agent_outcome"].return_values["output"]
        # print(f"result: {result}")
        # print(f"result of input: {bot_message['input']}")
        new_human_msg = HumanMessage(content=bot_message['input'])
        new_ai_msg = AIMessage(content=result)
        _chathistory.append(new_human_msg)
        _chathistory.append(new_ai_msg)
        # print(result)
        chat_history.append((message, result))
        # print(f"Chat History: {chat_history}")
        return "", chat_history


    textbox.submit(respond, [textbox, chatbot], [textbox, chatbot])
    chatbot.like(print_like_dislike, None, None)

    # demo.queue()

demo.launch(max_threads=8)
