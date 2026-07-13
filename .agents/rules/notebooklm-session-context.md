# NotebookLM Session Context Rule

The user maintains a NotebookLM notebook named "SNU AI Challenge - Project Settings" (id: `snu-ai-challenge-project-setti`, url: https://notebooklm.google.com/notebook/d115f01c-9e8e-4034-9fed-15c955b443fb) where they record project setup/config notes.

**Rule:** At the start of a session touching this project, if the `notebooklm` MCP tools are available, ask the notebook (`ask_question`, notebook_id `snu-ai-challenge-project-setti`) for the latest recorded setup notes before acting on setup/config-related requests. If the MCP server isn't available, just say so rather than silently skipping it.

**Why:** The user wants a lightweight way to carry project setup details across sessions beyond what's captured in this repo's own memory files.

**How to apply:** Relevant when the user asks about environment/training/experiment setup for this project, or says something like "check my notes" / "노트북 확인해줘". Not a hard gate on every single message — use judgment, and don't block on it if the request is clearly self-contained.