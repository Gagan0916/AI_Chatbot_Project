# AI Chatbot Development

Python | Generative AI | NLP

This project is a ChatGPT-style conversational AI chatbot with a browser interface, multi-chat sidebar, context-aware response handling, local NLP-style answers, and optional Generative AI integration. It runs with Python's standard library, so it is easy to demo without installing packages.

## Features

- Natural language chat interface
- ChatGPT-style layout with a conversation sidebar
- Multiple local chats saved in browser storage
- Context-aware conversation memory
- Streaming assistant replies for a ChatGPT-like typing experience
- Local NLP-style intent matching fallback
- Optional OpenAI-powered generative responses through `OPENAI_API_KEY`
- Runtime API Key button for temporary local testing without saving the key to files
- Prompt cards for project explanation, resume bullets, architecture, and code ideas
- Markdown-like message rendering with bullets and code blocks
- Copy, clear, delete, and regenerate controls
- Responsive web UI for desktop and mobile

## Run

```powershell
cd outputs\ai_chatbot
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

If port `8000` is busy, choose another port:

```powershell
$env:CHATBOT_PORT="8010"
python app.py
```

## Optional Generative AI Mode

Set an API key before running the app:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python app.py
```

Without an API key, the chatbot still works using the included local response engine.

You can also click `API Key` inside the app and paste a fresh key for the current running server only. That key is kept in memory and is not written to this project.

## Windows Shortcut

Double-click `run.bat` from the project folder, then open:

```text
http://127.0.0.1:8000
```

## GitHub

This folder is ready to push as a GitHub repository. Do not commit real API keys. Use `.env.example` as the safe template.

## Deploy On Render

1. Push this repository to GitHub.
2. Open Render and create a new Web Service from the GitHub repo.
3. Use:

```text
Build Command: empty
Start Command: python app.py
```

4. Add `OPENAI_API_KEY` as an environment variable in Render if you want real model responses.

The app reads Render's `PORT` automatically.

## Resume Description

Developed a conversational AI chatbot leveraging Python, Generative AI, and NLP for intelligent query handling. Built a polished web interface that supports natural language interaction, conversation memory, quick prompt actions, and context-aware response generation.
