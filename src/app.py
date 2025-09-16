import gradio as gr
from uuid import uuid4

from agent import build_agent
from retriever import build_retriever_from_cv
from models import get_llm_summarization

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    CombinedMemory
)

def get_hybrid_memory():
    # use llm to summarize older conversation
    llm = get_llm_summarization()
    # keep last 5 messages
    window_memory =ConversationBufferWindowMemory (
        k = 5,
        return_messages = True
    )
    #summarize old messages
    summary_memory = ConversationSummaryMemory(
        llm = llm,
        return_messages = True
    )

    #combine memories
    hybrid_memory = CombinedMemory(memories = [window_memory, summary_memory])
    return hybrid_memory



def get_session_memory(session_id : str):
    return get_hybrid_memory()

def setup_agent(cv_file):
    retriever = build_retriever_from_cv(cv_file)
    agent = build_agent(retriever)

    agent_with_memory = RunnableWithMessageHistory (
        agent,
        get_session_memory,
        input_messages_key="input",
        history_messages_key="history"
    )
    return agent_with_memory

def chat_with_agent(user_input, chat_history, session_id, agent_with_memory):
    response = agent_with_memory.invoke(
        {'input': user_input},
        config = {"configurable": {"session_id": session_id}}
    )
    return chat_history + [(user_input, response.content)]

agent_with_memory = None 

with gr.Blocks() as demo:
    gr.Markdown('CV-powered Assistant with Hybrid Memory')
    session_id = str(uuid4())

    with gr.Row():
        cv_upload = gr.File(label = 'Upload your CV (PDF)')
    
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox('Ask me anything...')
    clear = gr.Button('Clear conversation')
    state = gr.State([])

    # Initialize agent on CV upload
    def init_agent(file):
        global agent_with_memory
        agent_with_memory = setup_agent(file)
        return "CV uploaded. You can now start chatting!"

    cv_upload.upload(init_agent, inputs=cv_upload, outputs=None)

    # User sends message
    def respond(message, chat_history):
        global agent_with_memory
        updated_history = chat_with_agent(message, chat_history, session_id, agent_with_memory)
        return "", updated_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)   