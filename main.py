from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from nodes import run_agent_reasoning, tool_node

AGENT_REASON = 'agent_reason'
ACT = 'act'
LAST = -1

def should_continue(state: MessagesState)->str:
    if not state['messages'][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    ACT: ACT,
    END: END
})

flow.add_edge(START, AGENT_REASON)
flow.add_edge(ACT, AGENT_REASON)
flow.add_edge(ACT, END)

graph = flow.compile()


def main():
    print("Hello from react-agent-executor!")
    print(graph.invoke({
        "messages": [
            HumanMessage(content="what is weather/temperature in Tokyo ? list it and then triple it")
        ]
    }))


if __name__ == "__main__":
    main()
