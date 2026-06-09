from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", os.environ.get("CHATBOT_PORT", "8000")))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
RUNTIME_API_KEY = ""

SYSTEM_PROMPT = (
    "You are Nova, a ChatGPT-style AI assistant built for a Python Generative AI "
    "and NLP chatbot project. Be helpful, conversational, practical, and concise. "
    "Use the chat history for context. When useful, format answers with bullets, "
    "steps, or short code blocks."
)

SESSIONS: dict[str, list[dict[str, str]]] = {}

KNOWLEDGE_BASE = [
    {
        "topic": "generative ai",
        "keywords": {"generative", "ai", "llm", "model", "prompt", "response", "openai"},
        "answer": (
            "Generative AI lets a chatbot create flexible responses from natural language "
            "instead of choosing from fixed scripts. In this project, the assistant can use "
            "an API-backed model when `OPENAI_API_KEY` is set, and a local fallback when it is not."
        ),
    },
    {
        "topic": "nlp",
        "keywords": {"nlp", "intent", "entity", "language", "query", "processing", "understand"},
        "answer": (
            "NLP helps the chatbot understand the user's message. It can identify intent, "
            "match topics, extract useful terms, and decide how to respond before generating "
            "the final answer."
        ),
    },
    {
        "topic": "python",
        "keywords": {"python", "backend", "server", "api", "http", "code"},
        "answer": (
            "Python powers the backend server, chat session memory, API routes, local response "
            "logic, and optional Generative AI integration. This version uses the standard "
            "library so the demo runs without package installation."
        ),
    },
    {
        "topic": "context",
        "keywords": {"context", "memory", "history", "previous", "conversation", "follow"},
        "answer": (
            "Context-aware chat means the assistant stores recent messages and uses them for "
            "follow-up questions. That is why prompts like 'explain it simpler' or 'write "
            "that as resume bullets' can refer to earlier parts of the chat."
        ),
    },
]


def get_api_key() -> str:
    return RUNTIME_API_KEY or os.environ.get("OPENAI_API_KEY", "")


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def compact_history(history: list[dict[str, str]], limit: int = 18) -> list[dict[str, str]]:
    return history[-limit:]


def latest_user_messages(history: list[dict[str, str]], limit: int = 3) -> list[str]:
    return [turn["content"] for turn in history if turn["role"] == "user"][-limit:]


def match_knowledge(message: str) -> dict[str, Any] | None:
    words = set(normalize(message))
    best_item: dict[str, Any] | None = None
    best_score = 0

    for item in KNOWLEDGE_BASE:
        score = len(words & item["keywords"])
        if score > best_score:
            best_item = item
            best_score = score

    return best_item if best_score else None


def has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def generate_local_response(message: str, history: list[dict[str, str]]) -> str:
    clean = message.strip()
    lowered = clean.lower()

    if not clean:
        return "Ask me anything about the chatbot project, Python, NLP, or Generative AI."

    if has_any(lowered, ("hello", "hi", "hey")) and len(clean) <= 28:
        return (
            "Hi, I am Nova. I can help with project explanation, resume bullets, "
            "architecture, Python code, and chatbot improvement ideas."
        )

    if has_any(lowered, ("real", "chatgpt", "like chatgpt", "actual ai")):
        return (
            "To make this behave like a real ChatGPT-style assistant, connect an AI model by "
            "setting `OPENAI_API_KEY` before starting the app.\n\n"
            "The interface is already shaped like a real chat app. With the key enabled, the "
            "same chat screen will use model-generated responses instead of the local fallback."
        )

    if has_any(lowered, ("resume", "cv", "bullet", "experience")):
        return (
            "Resume-ready bullets:\n\n"
            "- Developed a conversational AI chatbot using Python, Generative AI, and NLP for intelligent query handling.\n"
            "- Built a ChatGPT-style web interface with multi-chat history, prompt suggestions, and context-aware replies.\n"
            "- Implemented session memory, local NLP fallback logic, and optional OpenAI API integration for dynamic response generation."
        )

    if has_any(lowered, ("architecture", "flow", "pipeline", "how it works", "working")):
        return (
            "Architecture:\n\n"
            "1. The user enters a message in the browser UI.\n"
            "2. JavaScript sends the message and chat ID to `/api/chat`.\n"
            "3. Python loads recent session history for context.\n"
            "4. If `OPENAI_API_KEY` exists, the app sends the prompt to the AI model.\n"
            "5. Otherwise, local NLP-style matching creates a useful fallback answer.\n"
            "6. The response returns to the UI and is saved in the active conversation."
        )

    if has_any(lowered, ("feature", "features", "functionality", "capabilities")):
        return (
            "Core features:\n\n"
            "- ChatGPT-style conversation layout\n"
            "- Multiple local chats in the sidebar\n"
            "- Context-aware session memory\n"
            "- Markdown-like formatting for bullets and code\n"
            "- Local NLP fallback responses\n"
            "- Optional OpenAI model integration\n"
            "- Copy, delete, and new-chat actions"
        )

    if has_any(lowered, ("improve", "enhance", "upgrade", "better")):
        return (
            "Best next upgrades:\n\n"
            "- Add real streaming token-by-token responses.\n"
            "- Store conversations in SQLite instead of browser memory.\n"
            "- Add document upload and retrieval-augmented generation.\n"
            "- Add user authentication.\n"
            "- Add feedback buttons for answer quality.\n"
            "- Deploy it with HTTPS so it can be shared as a real portfolio project."
        )

    if has_any(lowered, ("code", "sample", "example", "python file")):
        return (
            "Here is a small Python example for the chatbot API idea:\n\n"
            "```python\n"
            "def chatbot_reply(message, history):\n"
            "    context = history[-6:]\n"
            "    if 'resume' in message.lower():\n"
            "        return 'Here are resume-ready project bullets...'\n"
            "    return generate_ai_or_local_answer(message, context)\n"
            "```\n\n"
            "In the full app, the browser sends a message to `/api/chat`, and the Python server returns a JSON reply."
        )

    if has_any(lowered, ("summary", "summarize", "recap")):
        turns = latest_user_messages(history)
        if not turns:
            return "We have just started, so there is not much to summarize yet."
        return "Quick recap:\n\n" + "\n".join(f"- {turn}" for turn in turns)

    if has_any(lowered, ("previous", "last question", "what did i ask")):
        previous = latest_user_messages(history, limit=1)
        if previous:
            return f"Your last question was:\n\n> {previous[0]}"
        return "I do not have an earlier question in this chat yet."

    if history and lowered in {"more", "continue", "explain more", "tell me more"}:
        return (
            "A stronger chatbot project usually has three layers:\n\n"
            "- Interface layer: the chat screen, prompt input, sidebar, and message rendering.\n"
            "- Intelligence layer: NLP matching, model prompts, memory, and response generation.\n"
            "- Data layer: saved conversations, uploaded files, analytics, or knowledge base search."
        )

    item = match_knowledge(clean)
    if item:
        return item["answer"] + "\n\nAsk me to turn this into resume bullets, architecture, or code."

    return (
        "I can help with that. For this project, I can give you a clear answer as:\n\n"
        "- a short explanation\n"
        "- resume bullets\n"
        "- architecture steps\n"
        "- Python code\n"
        "- improvement ideas\n\n"
        "Tell me which format you want."
    )


def generate_openai_response(message: str, history: list[dict[str, str]]) -> str | None:
    api_key = get_api_key()
    if not api_key:
        return None

    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *compact_history(history),
            {"role": "user", "content": message},
        ],
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))

    return "\n".join(chunk for chunk in chunks if chunk).strip() or None


def stream_openai_response(message: str, history: list[dict[str, str]]):
    api_key = get_api_key()
    if not api_key:
        return

    payload = {
        "model": MODEL,
        "stream": True,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *compact_history(history),
            {"role": "user", "content": message},
        ],
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    yield delta


def local_stream_chunks(text: str):
    for chunk in re.findall(r"\S+\s*", text):
        yield chunk
        time.sleep(0.018)


def stream_chat_reply(message: str, session_id: str):
    history = SESSIONS.setdefault(session_id, [])
    parts: list[str] = []

    if get_api_key():
        try:
            for delta in stream_openai_response(message, history):
                parts.append(delta)
                yield delta
        except (urllib.error.URLError, TimeoutError, OSError):
            pass

    response = "".join(parts).strip()
    if not response:
        response = generate_local_response(message, history)
        for delta in local_stream_chunks(response):
            yield delta

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    SESSIONS[session_id] = compact_history(history, limit=24)


def chatbot_reply(message: str, session_id: str) -> str:
    history = SESSIONS.setdefault(session_id, [])
    response = generate_openai_response(message, history)
    if response is None:
        response = generate_local_response(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    SESSIONS[session_id] = compact_history(history, limit=24)
    return response


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)


def page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nova Chatbot</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #ffffff;
      --sidebar: #f5f6f7;
      --sidebar-hover: #e9ecef;
      --text: #1f2328;
      --muted: #6b7280;
      --line: #d9dee3;
      --soft-line: #edf0f2;
      --assistant: #ffffff;
      --user: #f1f8f5;
      --green: #10a37f;
      --green-dark: #0b7f62;
      --danger: #b42318;
      --shadow: 0 16px 44px rgba(27, 31, 35, 0.09);
      --composer-shadow: 0 -10px 30px rgba(27, 31, 35, 0.07);
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      height: 100%;
    }

    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }

    button,
    textarea {
      font: inherit;
    }

    button {
      cursor: pointer;
    }

    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: 292px minmax(0, 1fr);
      overflow: hidden;
    }

    .sidebar {
      min-width: 0;
      background: var(--sidebar);
      border-right: 1px solid var(--line);
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      overflow: hidden;
    }

    .side-top {
      padding: 14px;
      display: grid;
      gap: 10px;
    }

    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-width: 0;
    }

    .brand-mark {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      font-weight: 850;
    }

    .brand-avatar {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: #fff;
      background: var(--green);
      font-weight: 950;
      flex: 0 0 auto;
    }

    .status-pill {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--muted);
      padding: 7px 9px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .new-chat {
      height: 44px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-weight: 850;
    }

    .new-chat:hover,
    .chat-item:hover,
    .icon-button:hover,
    .prompt:hover {
      background: var(--sidebar-hover);
    }

    .search-area {
      padding: 0 14px 12px;
    }

    .search {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 11px;
      outline: none;
      background: #fff;
      color: var(--text);
    }

    .chat-list {
      overflow-y: auto;
      padding: 0 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .chat-item {
      min-height: 42px;
      border: 0;
      border-radius: 8px;
      background: transparent;
      color: var(--text);
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 8px 8px 8px 10px;
      text-align: left;
    }

    .chat-item.active {
      background: #fff;
      box-shadow: inset 0 0 0 1px var(--line);
    }

    .chat-title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
      font-weight: 720;
    }

    .delete-chat {
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: var(--muted);
      width: 28px;
      height: 28px;
      font-size: 16px;
    }

    .delete-chat:hover {
      background: #fff0ef;
      color: var(--danger);
    }

    .side-footer {
      border-top: 1px solid var(--line);
      padding: 12px 14px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    .main {
      min-width: 0;
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      background: var(--bg);
    }

    .topbar {
      height: 58px;
      border-bottom: 1px solid var(--soft-line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 20px;
    }

    .topbar-title {
      min-width: 0;
    }

    .topbar-title strong {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 15px;
    }

    .topbar-title span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }

    .topbar-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 0 0 auto;
    }

    .icon-button {
      height: 36px;
      min-width: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font-weight: 850;
      padding: 0 11px;
    }

    .conversation {
      overflow-y: auto;
      padding: 0;
      scroll-behavior: smooth;
    }

    .empty {
      width: min(820px, calc(100vw - 340px));
      min-height: calc(100vh - 230px);
      margin: 0 auto;
      padding: 58px 20px 30px;
      display: grid;
      align-content: center;
      gap: 26px;
    }

    .empty h1 {
      margin: 0;
      font-size: clamp(32px, 5vw, 46px);
      line-height: 1.08;
      text-align: center;
      letter-spacing: 0;
    }

    .prompt-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .prompt {
      min-height: 78px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 14px;
      text-align: left;
      display: grid;
      align-content: center;
      gap: 4px;
      box-shadow: 0 8px 18px rgba(27, 31, 35, 0.04);
    }

    .prompt strong {
      font-size: 14px;
    }

    .prompt span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }

    .messages {
      padding: 18px 0 28px;
    }

    .message-row {
      border-bottom: 1px solid var(--soft-line);
    }

    .message-row.user {
      background: var(--user);
    }

    .message-inner {
      width: min(820px, calc(100vw - 340px));
      margin: 0 auto;
      display: grid;
      grid-template-columns: 38px minmax(0, 1fr);
      gap: 14px;
      padding: 20px;
    }

    .avatar {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: #fff;
      font-size: 12px;
      font-weight: 950;
      background: var(--green);
    }

    .message-row.user .avatar {
      background: #3b3f45;
    }

    .message-body {
      min-width: 0;
      line-height: 1.68;
      font-size: 15px;
    }

    .message-body p {
      margin: 0 0 12px;
    }

    .message-body p:last-child {
      margin-bottom: 0;
    }

    .message-body ul,
    .message-body ol {
      margin: 0 0 12px 22px;
      padding: 0;
    }

    .message-body li {
      margin: 4px 0;
    }

    .message-body blockquote {
      margin: 0 0 12px;
      border-left: 3px solid var(--line);
      padding-left: 12px;
      color: var(--muted);
    }

    .message-body pre {
      margin: 12px 0;
      padding: 14px;
      overflow: auto;
      border-radius: 8px;
      background: #151718;
      color: #f4f7f8;
      font-size: 13px;
      line-height: 1.55;
    }

    .message-body code {
      font-family: "Cascadia Code", Consolas, "Liberation Mono", monospace;
    }

    .message-body :not(pre) > code {
      background: #eef1f3;
      border: 1px solid #dfe5e9;
      border-radius: 5px;
      padding: 2px 5px;
      color: #29313a;
      font-size: 0.92em;
    }

    .message-tools {
      margin-top: 10px;
      display: flex;
      gap: 8px;
      align-items: center;
    }

    .tiny-button {
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--muted);
      padding: 0 9px;
      font-size: 12px;
      font-weight: 850;
    }

    .tiny-button:hover {
      color: var(--text);
      background: var(--sidebar-hover);
    }

    .composer-wrap {
      background: linear-gradient(180deg, rgba(255, 255, 255, 0), #fff 22%);
      padding: 16px 20px 22px;
      box-shadow: var(--composer-shadow);
    }

    .composer {
      width: min(820px, calc(100vw - 340px));
      margin: 0 auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: var(--shadow);
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: end;
      gap: 8px;
      padding: 10px;
    }

    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 190px;
      resize: none;
      border: 0;
      outline: none;
      color: var(--text);
      padding: 12px 10px;
      line-height: 1.45;
    }

    .send {
      width: 44px;
      height: 44px;
      border: 0;
      border-radius: 8px;
      background: var(--green);
      color: #fff;
      font-weight: 950;
      font-size: 13px;
    }

    .send:disabled {
      background: #a8b2ba;
      cursor: wait;
    }

    .hint {
      width: min(820px, calc(100vw - 340px));
      margin: 8px auto 0;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }

    .typing {
      color: var(--muted);
      display: inline-flex;
      gap: 3px;
      align-items: center;
    }

    .dot {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--muted);
      animation: bounce 1s infinite ease-in-out;
    }

    .dot:nth-child(2) {
      animation-delay: 0.15s;
    }

    .dot:nth-child(3) {
      animation-delay: 0.3s;
    }

    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
      40% { transform: translateY(-4px); opacity: 1; }
    }

    @media (max-width: 900px) {
      .app {
        grid-template-columns: 1fr;
      }

      .sidebar {
        display: none;
      }

      .empty,
      .message-inner,
      .composer,
      .hint {
        width: min(820px, calc(100vw - 24px));
      }

      .topbar {
        padding: 0 12px;
      }
    }

    @media (max-width: 560px) {
      .prompt-grid {
        grid-template-columns: 1fr;
      }

      .message-inner {
        grid-template-columns: 32px minmax(0, 1fr);
        gap: 10px;
        padding: 16px 12px;
      }

      .avatar {
        width: 30px;
        height: 30px;
      }

      .composer-wrap {
        padding: 12px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar" aria-label="Chats">
      <div class="side-top">
        <div class="brand">
          <div class="brand-mark">
            <div class="brand-avatar">N</div>
            <span>Nova</span>
          </div>
          <div class="status-pill" id="mode">Local mode</div>
        </div>
        <button class="new-chat" type="button" id="new-chat">+ New chat</button>
      </div>

      <div class="search-area">
        <input class="search" id="search" placeholder="Search chats" autocomplete="off">
      </div>

      <div class="chat-list" id="chat-list"></div>

      <div class="side-footer">
        Python / Generative AI / NLP<br>
        Connect an API key for real model responses.
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div class="topbar-title">
          <strong id="active-title">New chat</strong>
          <span id="active-subtitle">Nova assistant</span>
        </div>
        <div class="topbar-actions">
          <button class="icon-button" type="button" id="set-api-key">API Key</button>
          <button class="icon-button" type="button" id="copy-conversation">Copy</button>
          <button class="icon-button" type="button" id="clear-conversation">Clear</button>
        </div>
      </header>

      <section class="conversation" id="conversation" aria-label="Conversation"></section>

      <section class="composer-wrap">
        <form class="composer" id="chat-form">
          <textarea id="message" placeholder="Message Nova" rows="1" autocomplete="off"></textarea>
          <button class="send" id="send" type="submit" aria-label="Send">Send</button>
        </form>
        <div class="hint" id="hint">Local fallback is active unless `OPENAI_API_KEY` is set before launch.</div>
      </section>
    </main>
  </div>

  <template id="empty-template">
    <div class="empty">
      <h1>What can I help with?</h1>
      <div class="prompt-grid">
        <button class="prompt" type="button" data-prompt="Explain this chatbot project like an interviewer is asking me">
          <strong>Explain the project</strong>
          <span>Clear interview-style answer</span>
        </button>
        <button class="prompt" type="button" data-prompt="Write resume bullets for this AI chatbot project">
          <strong>Resume bullets</strong>
          <span>Professional project points</span>
        </button>
        <button class="prompt" type="button" data-prompt="Show the architecture of this chatbot">
          <strong>Architecture</strong>
          <span>Frontend, backend, AI flow</span>
        </button>
        <button class="prompt" type="button" data-prompt="Give me Python code ideas for this chatbot">
          <strong>Code ideas</strong>
          <span>Implementation details</span>
        </button>
      </div>
    </div>
  </template>

  <script>
    const STORAGE_KEY = "nova-chatgpt-style-chats-v1";
    const chatList = document.querySelector("#chat-list");
    const conversation = document.querySelector("#conversation");
    const form = document.querySelector("#chat-form");
    const input = document.querySelector("#message");
    const send = document.querySelector("#send");
    const mode = document.querySelector("#mode");
    const hint = document.querySelector("#hint");
    const activeTitle = document.querySelector("#active-title");
    const activeSubtitle = document.querySelector("#active-subtitle");
    const search = document.querySelector("#search");
    const setApiKey = document.querySelector("#set-api-key");

    let chats = loadChats();
    let activeId = chats[0]?.id || "";
    let busy = false;
    if (!activeId) activeId = createChat(false);

    function uid() {
      return crypto.randomUUID ? crypto.randomUUID() : `chat-${Date.now()}-${Math.random()}`;
    }

    function loadChats() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        return Array.isArray(parsed) ? parsed : [];
      } catch {
        return [];
      }
    }

    function saveChats() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats.slice(0, 30)));
    }

    function createChat(render = true) {
      const chat = {
        id: uid(),
        title: "New chat",
        createdAt: Date.now(),
        messages: []
      };
      chats.unshift(chat);
      activeId = chat.id;
      saveChats();
      if (render) renderAll();
      return chat.id;
    }

    function activeChat() {
      let chat = chats.find(item => item.id === activeId);
      if (!chat) {
        activeId = createChat(false);
        chat = chats.find(item => item.id === activeId);
      }
      return chat;
    }

    function titleFromMessage(text) {
      const clean = text.replace(/\\s+/g, " ").trim();
      if (!clean) return "New chat";
      return clean.length > 34 ? clean.slice(0, 34) + "..." : clean;
    }

    function renderAll() {
      renderSidebar();
      renderConversation();
    }

    function renderSidebar() {
      const query = search.value.trim().toLowerCase();
      chatList.innerHTML = "";

      chats
        .filter(chat => !query || chat.title.toLowerCase().includes(query))
        .forEach(chat => {
          const item = document.createElement("button");
          item.className = `chat-item ${chat.id === activeId ? "active" : ""}`;
          item.type = "button";
          item.addEventListener("click", () => {
            activeId = chat.id;
            renderAll();
          });

          const title = document.createElement("span");
          title.className = "chat-title";
          title.textContent = chat.title;

          const del = document.createElement("button");
          del.className = "delete-chat";
          del.type = "button";
          del.textContent = "x";
          del.setAttribute("aria-label", "Delete chat");
          del.addEventListener("click", event => {
            event.stopPropagation();
            deleteChat(chat.id);
          });

          item.appendChild(title);
          item.appendChild(del);
          chatList.appendChild(item);
        });
    }

    function deleteChat(id) {
      chats = chats.filter(chat => chat.id !== id);
      fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: id })
      }).catch(() => {});

      if (!chats.length) {
        activeId = createChat(false);
      } else if (activeId === id) {
        activeId = chats[0].id;
      }

      saveChats();
      renderAll();
    }

    function renderConversation() {
      const chat = activeChat();
      activeTitle.textContent = chat.title;
      activeSubtitle.textContent = chat.messages.length ? `${chat.messages.length} messages` : "Nova assistant";
      conversation.innerHTML = "";

      if (!chat.messages.length) {
        const node = document.querySelector("#empty-template").content.cloneNode(true);
        conversation.appendChild(node);
        conversation.querySelectorAll("[data-prompt]").forEach(button => {
          button.addEventListener("click", () => submitMessage(button.dataset.prompt));
        });
        return;
      }

      const list = document.createElement("div");
      list.className = "messages";
      chat.messages.forEach((message, index) => list.appendChild(messageNode(message, index)));
      conversation.appendChild(list);
      conversation.scrollTop = conversation.scrollHeight;
    }

    function messageNode(message, index) {
      const row = document.createElement("article");
      row.className = `message-row ${message.role}`;

      const inner = document.createElement("div");
      inner.className = "message-inner";

      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = message.role === "user" ? "You" : "AI";

      const bodyWrap = document.createElement("div");
      const body = document.createElement("div");
      body.className = "message-body";
      body.innerHTML = message.pending && !message.text ? typingHtml() : renderMarkdown(message.text);
      bodyWrap.appendChild(body);

      if (!message.pending) {
        const tools = document.createElement("div");
        tools.className = "message-tools";

        const copy = document.createElement("button");
        copy.className = "tiny-button";
        copy.type = "button";
        copy.textContent = "Copy";
        copy.addEventListener("click", () => copyText(message.text, copy));
        tools.appendChild(copy);

        if (message.role === "assistant" && index === activeChat().messages.length - 1) {
          const again = document.createElement("button");
          again.className = "tiny-button";
          again.type = "button";
          again.textContent = "Regenerate";
          again.addEventListener("click", regenerateLast);
          tools.appendChild(again);
        }

        bodyWrap.appendChild(tools);
      }

      inner.appendChild(avatar);
      inner.appendChild(bodyWrap);
      row.appendChild(inner);
      return row;
    }

    function escapeHtml(text) {
      return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function renderInline(text) {
      return escapeHtml(text).replace(/`([^`]+)`/g, "<code>$1</code>");
    }

    function renderMarkdown(text) {
      const parts = text.split(/```/);
      let html = "";

      parts.forEach((part, index) => {
        if (index % 2 === 1) {
          const lines = part.replace(/^\\w+\\n/, "");
          html += `<pre><code>${escapeHtml(lines.trim())}</code></pre>`;
          return;
        }

        const block = part.trim();
        if (!block) return;

        const lines = block.split(/\\n+/);
        let listTag = "";
        lines.forEach(line => {
          if (/^\\s*[-*]\\s+/.test(line)) {
            if (listTag !== "ul") {
              if (listTag) html += `</${listTag}>`;
              html += "<ul>";
              listTag = "ul";
            }
            html += `<li>${renderInline(line.replace(/^\\s*[-*]\\s+/, ""))}</li>`;
          } else if (/^\\s*\\d+\\.\\s+/.test(line)) {
            if (listTag !== "ol") {
              if (listTag) html += `</${listTag}>`;
              html += "<ol>";
              listTag = "ol";
            }
            html += `<li>${renderInline(line.replace(/^\\s*\\d+\\.\\s+/, ""))}</li>`;
          } else {
            if (listTag) {
              html += `</${listTag}>`;
              listTag = "";
            }
            if (/^\\s*&gt;/.test(renderInline(line))) {
              html += `<blockquote>${renderInline(line.replace(/^\\s*>\\s?/, ""))}</blockquote>`;
            } else {
              html += `<p>${renderInline(line)}</p>`;
            }
          }
        });
        if (listTag) html += `</${listTag}>`;
      });

      return html || "<p></p>";
    }

    function typingHtml() {
      return '<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
    }

    function pushMessage(role, text, pending = false) {
      const chat = activeChat();
      chat.messages.push({ role, text, pending, at: Date.now() });
      if (role === "user" && chat.title === "New chat") {
        chat.title = titleFromMessage(text);
      }
      saveChats();
      renderAll();
    }

    function pendingAssistantMessage() {
      return activeChat().messages.findLast(item => item.pending);
    }

    function updatePendingAssistant(delta = "", done = false) {
      const pending = pendingAssistantMessage();
      if (!pending) return;
      pending.text += delta;
      if (done) pending.pending = false;
      saveChats();
      renderConversation();
    }

    async function streamAssistantReply(message, sessionId) {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId })
      });

      if (!response.ok || !response.body) throw new Error("Stream failed");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\\n\\n");
        buffer = events.pop() || "";

        for (const event of events) {
          const line = event.split("\\n").find(item => item.startsWith("data:"));
          if (!line) continue;

          const payload = JSON.parse(line.slice(5).trim());
          if (payload.delta) updatePendingAssistant(payload.delta, false);
          if (payload.done) updatePendingAssistant("", true);
        }
      }

      updatePendingAssistant("", true);
    }

    async function submitMessage(text) {
      const value = text.trim();
      if (!value || busy) return;

      busy = true;
      send.disabled = true;
      input.value = "";
      autoSize();

      const chat = activeChat();
      pushMessage("user", value);
      pushMessage("assistant", "", true);

      try {
        await streamAssistantReply(value, chat.id);
      } catch {
        const pending = pendingAssistantMessage();
        if (pending) {
          pending.text = "I could not process that request. Please try again.";
          pending.pending = false;
        }
      } finally {
        busy = false;
        send.disabled = false;
        saveChats();
        renderAll();
        input.focus();
      }
    }

    async function regenerateLast() {
      const chat = activeChat();
      const lastUser = [...chat.messages].reverse().find(item => item.role === "user");
      if (!lastUser || busy) return;
      while (chat.messages.length && chat.messages[chat.messages.length - 1].role === "assistant") {
        chat.messages.pop();
      }
      saveChats();
      busy = true;
      send.disabled = true;
      pushMessage("assistant", "", true);

      try {
        await streamAssistantReply(lastUser.text, chat.id);
      } catch {
        const pending = pendingAssistantMessage();
        if (pending) {
          pending.text = "I could not regenerate that response. Please try again.";
          pending.pending = false;
        }
      } finally {
        busy = false;
        send.disabled = false;
        saveChats();
        renderAll();
      }
    }

    async function copyText(text, button) {
      try {
        await navigator.clipboard.writeText(text);
        const original = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => button.textContent = original, 900);
      } catch {
        button.textContent = "Failed";
      }
    }

    function transcript() {
      return activeChat().messages
        .filter(message => !message.pending)
        .map(message => `${message.role.toUpperCase()}: ${message.text}`)
        .join("\\n\\n");
    }

    function autoSize() {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 190) + "px";
    }

    form.addEventListener("submit", event => {
      event.preventDefault();
      submitMessage(input.value);
    });

    input.addEventListener("input", autoSize);
    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitMessage(input.value);
      }
    });

    document.querySelector("#new-chat").addEventListener("click", () => createChat(true));
    document.querySelector("#copy-conversation").addEventListener("click", event => {
      copyText(transcript(), event.currentTarget);
    });
    document.querySelector("#clear-conversation").addEventListener("click", async () => {
      const chat = activeChat();
      chat.messages = [];
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: chat.id })
      }).catch(() => {});
      saveChats();
      renderAll();
    });
    search.addEventListener("input", renderSidebar);

    async function refreshStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();
        mode.textContent = data.ai_enabled ? `AI: ${data.model}` : "Local mode";
        hint.textContent = data.ai_enabled
          ? `Using ${data.model} for model-generated responses.`
          : "Local fallback is active unless `OPENAI_API_KEY` is set before launch.";
      } catch {}
    }

    setApiKey.addEventListener("click", async () => {
      const apiKey = prompt("Paste a fresh OpenAI API key. It stays only in this running local server.");
      if (!apiKey) return;

      const response = await fetch("/api/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey })
      });

      if (response.ok) {
        await refreshStatus();
      }
    });

    refreshStatus();

    renderAll();
    autoSize();
  </script>
</body>
</html>"""


class ChatbotHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_html(page())
            return

        if self.path == "/api/status":
            self.send_json(
                {
                    "ai_enabled": bool(get_api_key()),
                    "model": MODEL,
                    "sessions": len(SESSIONS),
                    "runtime_key": bool(RUNTIME_API_KEY),
                }
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Page not found")

    def do_POST(self) -> None:
        if self.path == "/api/chat/stream":
            payload = self.read_json()
            if payload is None:
                return

            message = str(payload.get("message", ""))
            session_id = str(payload.get("session_id", "default"))[:120] or "default"
            self.send_stream_start()
            try:
                for delta in stream_chat_reply(message, session_id):
                    self.send_stream_data({"delta": delta})
                self.send_stream_data({"done": True, "created_at": int(time.time())})
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        if self.path == "/api/chat":
            payload = self.read_json()
            if payload is None:
                return

            message = str(payload.get("message", ""))
            session_id = str(payload.get("session_id", "default"))[:120] or "default"
            reply = chatbot_reply(message, session_id)
            self.send_json({"reply": reply, "created_at": int(time.time())})
            return

        if self.path == "/api/reset":
            payload = self.read_json()
            if payload is None:
                return

            session_id = str(payload.get("session_id", "default"))[:120] or "default"
            reset_session(session_id)
            self.send_json({"ok": True})
            return

        if self.path == "/api/key":
            payload = self.read_json()
            if payload is None:
                return

            global RUNTIME_API_KEY
            api_key = str(payload.get("api_key", "")).strip()
            if api_key:
                RUNTIME_API_KEY = api_key
                self.send_json({"ok": True, "ai_enabled": True, "model": MODEL})
            else:
                RUNTIME_API_KEY = ""
                self.send_json({"ok": True, "ai_enabled": bool(os.environ.get("OPENAI_API_KEY")), "model": MODEL})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def read_json(self) -> dict[str, Any] | None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
            return None

    def send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_stream_start(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def send_stream_data(self, payload: dict[str, Any]) -> None:
        encoded = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(encoded)
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ChatbotHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"Nova Chatbot running at {url}")
    print("Set OPENAI_API_KEY before launch to enable real model responses.")
    server.serve_forever()


if __name__ == "__main__":
    main()
