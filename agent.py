"""
AutoStream Conversational AI Agent
Built with LangGraph + Claude 3 Haiku (via Anthropic API)

Capabilities:
- Intent classification (greeting / product inquiry / high-intent lead)
- RAG-powered knowledge retrieval from local JSON knowledge base
- Lead capture tool triggered only when high intent is confirmed
- Multi-turn state management via LangGraph StateGraph
"""

import json
import os
import re
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ── Knowledge Base ──────────────────────────────────────────────────────────

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "autostream_kb.json")

with open(KB_PATH, "r") as f:
    KNOWLEDGE_BASE = json.load(f)

def build_kb_context() -> str:
    """Flatten knowledge base into a readable context string for the LLM."""
    lines = [
        f"Company: {KNOWLEDGE_BASE['company']}",
        f"About: {KNOWLEDGE_BASE['description']}",
        "",
        "== PRICING PLANS ==",
    ]
    for plan in KNOWLEDGE_BASE["plans"]:
        lines.append(f"\n{plan['name']} — {plan['price']}")
        for feat in plan["features"]:
            lines.append(f"  • {feat}")

    lines.append("\n== COMPANY POLICIES ==")
    for policy in KNOWLEDGE_BASE["policies"]:
        lines.append(f"\n{policy['topic']}: {policy['detail']}")

    lines.append("\n== FAQs ==")
    for faq in KNOWLEDGE_BASE["faqs"]:
        lines.append(f"\nQ: {faq['question']}\nA: {faq['answer']}")

    return "\n".join(lines)

KB_CONTEXT = build_kb_context()

# ── Tool: Mock Lead Capture ─────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API call that simulates saving a lead to CRM."""
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"

# ── State Schema ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: str                                # current classified intent
    collecting_lead: bool                      # are we in lead-collection mode?
    lead_data: dict                            # name / email / platform collected so far
    lead_captured: bool                        # has the lead been saved?

# ── LLM Setup ──────────────────────────────────────────────────────────────

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=api_key)

# ── Intent Classification Node ──────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video editing platform.
Classify the user's latest message into EXACTLY one of:
  - GREETING       : casual hello, how are you, small talk
  - PRODUCT_INQUIRY: questions about features, pricing, plans, policies, FAQs
  - HIGH_INTENT    : user explicitly wants to sign up, try, purchase, or start a plan

Reply with ONLY the label (one word, uppercase). No explanation."""

def classify_intent(state: AgentState) -> AgentState:
    llm = get_llm()
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    response = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=last_human),
    ])
    raw = response.content.strip().upper()
    # Normalise — accept partial matches
    if "HIGH" in raw:
        intent = "HIGH_INTENT"
    elif "PRODUCT" in raw or "INQUIRY" in raw:
        intent = "PRODUCT_INQUIRY"
    else:
        intent = "GREETING"
    return {**state, "intent": intent}

# ── Router ──────────────────────────────────────────────────────────────────

def route(state: AgentState) -> Literal["collect_lead", "respond"]:
    if state.get("collecting_lead"):
        return "collect_lead"
    if state["intent"] == "HIGH_INTENT":
        return "collect_lead"
    return "respond"

# ── General Response Node (RAG) ─────────────────────────────────────────────

RESPONSE_SYSTEM = f"""You are Alex, a friendly and knowledgeable sales assistant for AutoStream —
an AI-powered video editing SaaS for content creators.

Use ONLY the knowledge base below to answer product/pricing questions.
Be concise, warm, and helpful. If a user's question is not covered, say so honestly.

{KB_CONTEXT}

Rules:
- Never make up features or prices.
- If intent seems to be shifting toward sign-up, gently encourage them.
- Keep responses under 120 words unless detail is explicitly requested.
"""

def respond(state: AgentState) -> AgentState:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=RESPONSE_SYSTEM),
        *state["messages"],
    ])
    return {**state, "messages": [response]}

# ── Lead Collection Node ────────────────────────────────────────────────────

LEAD_SYSTEM = """You are Alex from AutoStream. A user has shown interest in signing up.
Your job is to collect their details ONE AT A TIME in this order:
  1. Full name
  2. Email address
  3. Creator platform (e.g., YouTube, Instagram, TikTok)

Current collected data: {lead_data}

Rules:
- Only ask for the NEXT missing field.
- If all three are collected, confirm with a warm closing message and say LEAD_COMPLETE.
- Do NOT ask for a field that is already collected.
- Be friendly and conversational.
"""

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PLATFORM_KEYWORDS = ["youtube", "instagram", "tiktok", "twitter", "x", "facebook",
                     "twitch", "linkedin", "snapchat", "pinterest", "vimeo"]

def extract_lead_fields(text: str, current: dict) -> dict:
    """Best-effort extraction of lead fields from user message."""
    updated = dict(current)
    lower = text.lower().strip()

    # Email
    if "email" not in updated:
        match = EMAIL_RE.search(text)
        if match:
            updated["email"] = match.group()

    # Platform
    if "platform" not in updated:
        for kw in PLATFORM_KEYWORDS:
            if kw in lower:
                updated["platform"] = kw.capitalize()
                break

    # Name — collect if email & platform are filled, or if nothing else fits
    if "name" not in updated:
        # Heuristic: if no email found in this message and no platform keyword,
        # treat the message as a name (only if it looks like a name: 1-4 words, no @)
        if "@" not in text and not any(kw in lower for kw in PLATFORM_KEYWORDS):
            words = text.strip().split()
            if 1 <= len(words) <= 4:
                updated["name"] = text.strip().title()

    return updated

def collect_lead(state: AgentState) -> AgentState:
    llm = get_llm()
    lead_data = state.get("lead_data", {})

    # Try to extract info from latest human message
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    lead_data = extract_lead_fields(last_human, lead_data)

    # Check completeness
    all_collected = all(k in lead_data for k in ("name", "email", "platform"))

    if all_collected and not state.get("lead_captured"):
        # Fire the tool
        result = mock_lead_capture(lead_data["name"], lead_data["email"], lead_data["platform"])
        closing = (
            f"🎉 You're all set, {lead_data['name']}! "
            f"Our team will reach out to {lead_data['email']} shortly to get your AutoStream Pro "
            f"account up and running for your {lead_data['platform']} channel. "
            f"Welcome aboard! 🚀"
        )
        return {
            **state,
            "messages": [AIMessage(content=closing)],
            "lead_data": lead_data,
            "lead_captured": True,
            "collecting_lead": False,
        }

    # Ask for the next missing field
    prompt = LEAD_SYSTEM.format(lead_data=json.dumps(lead_data))
    response = llm.invoke([
        SystemMessage(content=prompt),
        *state["messages"],
    ])
    return {
        **state,
        "messages": [response],
        "lead_data": lead_data,
        "collecting_lead": True,
    }

# ── Graph Assembly ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("respond", respond)
    graph.add_node("collect_lead", collect_lead)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route)
    graph.add_edge("respond", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()

# ── CLI Runner ──────────────────────────────────────────────────────────────

def run_cli():
    print("=" * 60)
    print("  AutoStream AI Agent  (type 'quit' to exit)")
    print("=" * 60)

    compiled = build_graph()
    state: AgentState = {
        "messages": [],
        "intent": "GREETING",
        "collecting_lead": False,
        "lead_data": {},
        "lead_captured": False,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Agent: Thanks for chatting! Have a great day. 👋")
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = compiled.invoke(state)

        last_ai = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            "Sorry, I couldn't generate a response."
        )
        print(f"\nAgent: {last_ai}")

        if state.get("lead_captured"):
            print("\n[Session complete — lead successfully captured]")
            break

if __name__ == "__main__":
    run_cli()
