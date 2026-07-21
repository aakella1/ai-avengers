"""A chat UI for the agent, in about 150 lines of Streamlit.

Nothing about the agent changes here. This file imports the same build_agent()
and the same TOOLS as agent.py — it only swaps the terminal for a browser. That
is worth saying out loud in class: the agent is a library, and the interface is
a detail on top of it.

The one thing this UI adds that the terminal cannot: tool calls render as they
happen, with the arguments the model chose. Students can watch it decide.

Run:
    streamlit run chat_app.py
"""

import asyncio

import streamlit as st

import tools_rag
from agent import MODEL_ID, build_agent

st.set_page_config(page_title="Strands Agent Demo", page_icon="🤖", layout="centered")


def new_agent():
    """A fresh agent, with tool activity silenced so we can render it ourselves."""
    # Passing callback_handler=None stops the SDK printing to the terminal.
    # It must be set at construction — assigning afterwards raises.
    return build_agent(callback_handler=None)


# Streamlit re-runs this whole file on every interaction, so anything that must
# survive between turns lives in session_state. The agent keeps its own message
# history internally, which is what gives us multi-turn memory for free.
if "agent" not in st.session_state:
    st.session_state.agent = new_agent()
if "history" not in st.session_state:
    st.session_state.history = []


async def stream_reply(prompt: str, text_box, tool_box) -> tuple[str, list[dict]]:
    """Consume the agent's event stream, updating the page as events arrive."""
    text = ""
    tools: list[dict] = []

    async for event in st.session_state.agent.stream_async(prompt):
        tool_use = event.get("current_tool_use") or {}

        # Tool events repeat as the arguments stream in, so key on toolUseId
        # to avoid showing the same call several times.
        if tool_use.get("name"):
            existing = next(
                (t for t in tools if t["id"] == tool_use.get("toolUseId")), None
            )
            if existing is None:
                tools.append(
                    {
                        "id": tool_use.get("toolUseId"),
                        "name": tool_use["name"],
                        "input": tool_use.get("input", ""),
                    }
                )
            else:
                existing["input"] = tool_use.get("input", existing["input"])
            render_tools(tool_box, tools, done=False)

        if "data" in event:
            text += event["data"]
            text_box.markdown(text)

    render_tools(tool_box, tools, done=True)
    return text, tools


def render_tools(box, tools: list[dict], done: bool) -> None:
    """Show which tools ran, and with what arguments."""
    if not tools:
        return

    label = (
        f"🔧 Used {len(tools)} tool{'s' if len(tools) != 1 else ''}"
        if done
        else f"🔧 Running {tools[-1]['name']}..."
    )
    with box.container():
        with st.expander(label, expanded=not done):
            for i, tool in enumerate(tools, start=1):
                st.markdown(f"**{i}. `{tool['name']}`**")
                if tool["input"]:
                    st.code(str(tool["input"]), language="json")


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Agent")
    st.caption(f"Model: `{MODEL_ID}`")

    tool_names = [
        t.tool_name
        for t in st.session_state.agent.tool_registry.registry.values()
    ]
    st.caption(f"{len(tool_names)} tools available")
    for name in tool_names:
        st.markdown(f"- `{name}`")

    st.divider()
    st.header("RAG index")

    indexes = tools_rag.list_indexes()
    if indexes:
        # Switching here changes which embedding model answers document
        # questions — a nice thing to flip mid-demo.
        choice = st.selectbox(
            "Used by `search_documents`",
            indexes,
            index=indexes.index(tools_rag.current_index())
            if tools_rag.current_index() in indexes
            else 0,
        )
        if choice != tools_rag.current_index():
            tools_rag.set_index(choice)
            st.caption("Switched. Loads on the next document search.")
    else:
        st.warning("No index found. Build one in the RAG/ folder first.")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.agent = new_agent()
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------- main

st.title("🤖 Agent chat")
st.caption(
    "Ask about the weather, do some maths, or ask about Chiranjeevi or "
    "Balakrishna to search the private PDFs."
)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn.get("tools"):
            with st.expander(f"🔧 Used {len(turn['tools'])} tools"):
                for i, tool in enumerate(turn["tools"], start=1):
                    st.markdown(f"**{i}. `{tool['name']}`**")
                    if tool["input"]:
                        st.code(str(tool["input"]), language="json")
        st.markdown(turn["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tool_box = st.empty()
        text_box = st.empty()
        try:
            reply, tools = asyncio.run(stream_reply(prompt, text_box, tool_box))
        except Exception as exc:  # noqa: BLE001 - show the failure in the UI
            reply, tools = f"⚠️ Something went wrong: `{exc}`", []
            text_box.markdown(reply)

    st.session_state.history.append(
        {"role": "assistant", "content": reply, "tools": tools}
    )
