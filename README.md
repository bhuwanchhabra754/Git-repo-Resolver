🚀 Git Repo Resolver

An AI-powered tool that analyzes GitHub repositories and answers questions about the codebase using embeddings, vector search, and Large Language Models (LLMs).

📌 Overview

Git Repo Resolver allows users to:

🔍 Analyze any public GitHub repository

🧠 Convert code into embeddings using SentenceTransformers

📦 Store embeddings in ChromaDB

💬 Ask natural language questions about the repository

🤖 Generate intelligent answers using Groq / OpenAI LLM

🎨 Interact through a simple Streamlit UI

🏗️ Architecture

Clone GitHub repository

Extract and chunk code files

Generate embeddings (all-MiniLM-L6-v2)

Store vectors in ChromaDB

Perform similarity search

Send relevant context to LLM

Display response in Streamlit UI

🛠️ Tech Stack

Python

Streamlit

ChromaDB

SentenceTransformers

Groq / OpenAI API

GitPython
