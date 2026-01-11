from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def should_continue(state: AgentState):
    last_msg = state['messages'][-1]
    has_tool_calls = hasattr(last_msg, 'tool_calls') and bool(getattr(last_msg, 'tool_calls', None))
    return "retriever_agent" if has_tool_calls else "end"

def call_llm(state: AgentState, llm, tools_dict, system_prompt: str):
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    bound_llm = llm.bind_tools(list(tools_dict.values()))
    message = bound_llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState, tools_dict):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        args = t.get("args", {})
        query = args.get("query") or args.get("question") or args.get("input") if isinstance(args, dict) else str(args)
        tool_name = t.get("name", "")
        result = tools_dict[tool_name].invoke(query) if tool_name in tools_dict else "Invalid tool name."
        results.append(ToolMessage(tool_call_id=t.get('id'), name=tool_name, content=str(result)))
    return {'messages': results}

def build_graph(llm, tools_dict, system_prompt):
    graph = StateGraph(AgentState)
    graph.add_node("llm", lambda state: call_llm(state, llm, tools_dict, system_prompt))
    graph.add_node("retriever_agent", lambda state: take_action(state, tools_dict))
    graph.add_conditional_edges("llm", should_continue, {"retriever_agent": "retriever_agent", "end": END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    return graph.compile()
