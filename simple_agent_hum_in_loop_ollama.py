import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch

# Load environment variables (для Tavily API ключа)
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")


# ============================================
# ВЫБОР МОДЕЛИ OLLAMA
# ============================================
# ВАЖНО: не все модели поддерживают bind_tools()!
# Лучшие для tools: llama3.1, llama3.2, qwen3
# llama3.2:latest (2 GB) — быстрая
# Для лучшей поддержки tools: llama3.1:latest (4.9 GB)
llm_name = "llama3.1:latest"

model = ChatOllama(
    model=llm_name,
    temperature=0.7,
    timeout=120,
)


# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# create tools
tool = TavilySearch(max_results=2)
tools = [tool]

model_with_tools = model.bind_tools(tools)

import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition


def bot(state: State):
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}


# instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

graph_builder.add_node("bot", bot)

# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# ADD MEMORY NODE
from langgraph.checkpoint.sqlite import SqliteSaver

# memory = SqliteSaver.from_conn_string(":memory:")
with SqliteSaver.from_conn_string(":memory:") as memory:
    # STEP 5: Compile the graph
    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["tools"],
    )

    # MEMORY CODE CONTINUES ===
    # PICK A THREAD FIRST
    config = {
        "configurable": {"thread_id": 1}
    }

    user_input = "I'm learning about astrology. Could you do some research on it for me?"

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # inspect the state
    snapshot = graph.get_state(config)
    next_step = snapshot.next

    print("===>>>", next_step)


    existing_message = snapshot.values["messages"][-1]
    all_tools = existing_message.tool_calls

    print("tools to be called::", all_tools)

    # Continue the conversation passing None to say continue
    events = graph.stream(None, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()