"""
AutoStream — WhatsApp Webhook Server
Integrates the LangGraph agent with the WhatsApp Business Cloud API.

Setup:
    pip install fastapi uvicorn httpx
    export ANTHROPIC_API_KEY=sk-ant-...
    export WA_VERIFY_TOKEN=my_secret
    export WA_ACCESS_TOKEN=EAAxxxxxx
    export WA_PHONE_NUMBER_ID=1234567890

Run:
    uvicorn whatsapp_webhook:app --port 8000
Then expose with ngrok:
    ngrok http 8000
"""

import os
import json
import httpx
from fastapi import FastAPI, Request, Response
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_graph, AgentState

app = FastAPI(title="AutoStream WhatsApp Webhook")
compiled = build_graph()

# In-memory session store — swap for Redis in production
sessions: dict[str, AgentState] = {}

VERIFY_TOKEN = os.environ.get("WA_VERIFY_TOKEN", "autostream_verify_token")
WA_TOKEN = os.environ.get("WA_ACCESS_TOKEN", "")
PHONE_ID = os.environ.get("WA_PHONE_NUMBER_ID", "")
WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_ID}/messages"


# ── Webhook Verification (GET) ──────────────────────────────────────────────

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta calls this endpoint to verify the webhook URL.
    We must echo back hub.challenge if the verify token matches.
    """
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print(f"[Webhook] Verified successfully ✅")
        return Response(content=challenge, media_type="text/plain")

    print(f"[Webhook] Verification failed ❌ — token mismatch")
    return Response(status_code=403, content="Forbidden")


# ── Inbound Message Handler (POST) ─────────────────────────────────────────

@app.post("/webhook")
async def receive_message(request: Request):
    """
    Meta sends inbound WhatsApp messages here as POST requests.
    We invoke the LangGraph agent and reply via the Cloud API.
    """
    body = await request.json()

    try:
        # Navigate the WhatsApp payload structure
        value = body["entry"][0]["changes"][0]["value"]
        msg = value["messages"][0]
        from_number: str = msg["from"]
        user_text: str = msg["text"]["body"]
        print(f"[{from_number}] User: {user_text}")
    except (KeyError, IndexError, TypeError):
        # Not a text message or unexpected shape — safely ignore
        return {"status": "ignored"}

    # Retrieve or create session state for this phone number
    state: AgentState = sessions.get(from_number, {
        "messages": [],
        "intent": "GREETING",
        "collecting_lead": False,
        "lead_data": {},
        "lead_captured": False,
    })

    # Append user message and invoke the agent
    state["messages"] = state["messages"] + [HumanMessage(content=user_text)]
    state = compiled.invoke(state)
    sessions[from_number] = state  # Persist updated state

    # Extract the latest AI reply
    reply = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        "Sorry, something went wrong. Please try again."
    )
    print(f"[{from_number}] Agent: {reply}")

    # Send reply back to the user via WhatsApp Cloud API
    await send_whatsapp_message(from_number, reply)

    # Clean up session after successful lead capture
    if state.get("lead_captured"):
        sessions.pop(from_number, None)

    return {"status": "ok"}


# ── Helper: Send WhatsApp Message ───────────────────────────────────────────

async def send_whatsapp_message(to: str, text: str) -> None:
    """POST a text message to a WhatsApp number via the Cloud API."""
    if not WA_TOKEN or not PHONE_ID:
        print("[Warning] WA_ACCESS_TOKEN or WA_PHONE_NUMBER_ID not set — skipping send")
        return

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            WA_API_URL,
            headers={
                "Authorization": f"Bearer {WA_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code != 200:
            print(f"[WhatsApp API Error] {response.status_code}: {response.text}")


# ── Health Check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "AutoStream AI", "sessions_active": len(sessions)}
