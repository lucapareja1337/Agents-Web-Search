# Scientific Article Search Agent using CrewAI

This project leverages **CrewAI** to orchestrate automated web searches focused on scientific articles. It uses intelligent agents to search, validate, and identify academic content across the internet, with a special emphasis on open-access sources.

## 🔍 Overview

The agent architecture is designed to:
1. Search for relevant scientific articles using the **Arxiv** platform.
2. Perform a web search (via **Tavily**, with a focus on Google results).
3. Identify and verify whether the results truly represent scientific articles.

The workflow is powered by large language models (**LLMs**) through APIs from **Groq** and **NVIDIA**, particularly using the **LLMA-70b** model for high-quality natural language reasoning.

## 🚀 Technologies Used

- **CrewAI** – Agent orchestration
- **Tavily API** – Web search integration (with emphasis on Google)
- **Groq & NVIDIA APIs** – Access to LLMA-70b and other advanced LLMs
- **Arxiv API** – Open-access scientific article search

## ⚙️ Setup Instructions

1. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
2. **Install dependencies:**
    
    pip install -r requirements.txt

3. **OBS!**

    ⚠️ Note: Ensure your NVIDIA API key is functioning properly.
    There have been occasional issues with connectivity.
    If the NVIDIA API fails, feel free to replace it with another LLM API of your choice (e.g., OpenAI, Mistral, etc.).
