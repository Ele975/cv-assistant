# import gradio as gr
# from uuid import uuid4

# from agent import build_agent
# from retriever import build_retriever_from_cv
# from models import get_llm_summarization

# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.memory import (
#     ConversationBufferWindowMemory,
#     ConversationSummaryMemory,
#     CombinedMemory,
#     ConversationBufferMemory
# )
# from langchain_core.chat_history import InMemoryChatMessageHistory

# session_histories = {}

# def get_hybrid_memory():
#     llm = get_llm_summarization()

#     # window_memory = ConversationBufferWindowMemory(
#     #     k=5,
#     #     memory_key="window_history",
#     #     input_key = "input_window",
#     #     return_messages=True
#     # )

#     summary_memory = ConversationSummaryMemory(
#         llm=llm,
#         # input_key = "input_summary",
#         # memory_key="summary_history",
#         return_messages=True
#     )

#     buffer_memory = ConversationBufferMemory(return_messages=True)
#     buffer_memory.chat_memory.messages = []  # Optional init
#     buffer_memory.summary_memory = summary_memory  # Store it for later use if needed
#     return buffer_memory

#     # return CombinedMemory(memories=[window_memory, summary_memory])


# def get_session_memory(session_id: str):
#     # combined_memory = get_hybrid_memory()
#     # buffer_memory = ConversationBufferMemory(return_messages = True)
#     # buffer_memory.chat_memory.messages = []
#     # buffer_memory.combined_memory = combined_memory
#     # return buffer_memory
#     # return get_hybrid_memory()
#     if session_id not in session_histories:
#         session_histories[session_id] = InMemoryChatMessageHistory()
#     return session_histories[session_id]


# def setup_agent(cv_file):
#     retriever = build_retriever_from_cv(cv_file)
#     agent = build_agent(retriever)

#     return RunnableWithMessageHistory(
#         agent,
#         get_session_memory,
#         input_messages_key="input",
#         history_messages_key="chat_history"
#     )


# def chat_with_agent(user_input, chat_history, session_id, agent_with_memory):
#     chat_history.append({"role": "user", "content": user_input})
#     print(f"💬 Invoking agent with input: {user_input}", flush=True)
#     response = agent_with_memory.invoke(
#         {"input": user_input},
#         config={"configurable": {"session_id": session_id}}
#     )
#     print("🧠 Raw response:", response, flush=True)

#     chat_history.append({"role": "assistant", "content": response.content})
#     return chat_history


# with gr.Blocks() as demo:
#     gr.Markdown("CV-powered Assistant with Hybrid Memory")
#     session_id = str(uuid4())

#     with gr.Row():
#         cv_upload = gr.File(label="Upload your CV (PDF)")
#         status = gr.Textbox(label="Status", interactive=False)

#     chatbot = gr.Chatbot(type="messages")
#     msg = gr.Textbox(placeholder="Type your question here...")
#     clear = gr.Button("Clear conversation")

#     # Keep the agent in state instead of using globals
#     agent_state = gr.State(None)

#     def init_agent(file):
#         print("⚙️ Setting up agent...", flush=True)
#         agent_with_memory = setup_agent(file)
#         print("✅ Agent created:", type(agent_with_memory), flush=True)
#         return agent_with_memory, "✅ CV uploaded. You can now start chatting!"

#     cv_upload.upload(
#         init_agent,
#         inputs=cv_upload,
#         outputs=[agent_state, status]  # update both state and status box
#     )

#     def respond(message, chat_history, agent_with_memory):
#         print("📩 Respond called:", message, flush=True)
#         print("🤖 Agent with memory is:", type(agent_with_memory), flush=True)
#         if agent_with_memory is None:
#             chat_history.append({"role": "system", "content": "Please upload a CV first."})
#             print('Chat history no agent:', chat_history)
#             return "", chat_history
#         try:
#             updated_history = chat_with_agent(message, chat_history, session_id, agent_with_memory)
#             print('Chat history agent:', updated_history)
#             return "", updated_history
#         except Exception as e:
#             chat_history.append({"role": "system", "content": f"Error: {str(e)}"})
#             print('Exception chat history:', chat_history)
#             return "", chat_history

#     msg.submit(respond, [msg, chatbot, agent_state], [msg, chatbot])
#     clear.click(lambda: [], None, chatbot, queue=False)

# demo.launch(server_name="0.0.0.0", server_port=7860)
