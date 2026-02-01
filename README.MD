====================================================
Restaurant AI Assistant (FastAPI + Streamlit + LangGraph + ChromaDB)
====================================================

Project Overview
----------------
This project is a Restaurant AI Assistant Application that combines AI, a Vector Database, FastAPI, and Streamlit. 
It allows users to upload restaurant data files and then interact with an AI-powered assistant that can:
- Suggest restaurants
- Show available options
- Guide users with booking suggestions (demo only, no real booking)

The application uses LangGraph to manage workflow, ChromaDB to store and search restaurant data, 
and OpenRouter AI (DeepSeek model) to generate intelligent responses.

----------------------------------------------------
How It Works
------------
1. Upload Restaurant Data
   - You can upload .csv, .txt, or .do files containing restaurant details.
   - Data is parsed and stored in ChromaDB (vector database) for semantic search.

2. Ask Queries
   - Example queries:
     - "Suggest Italian restaurants for dinner"
     - "Book a table for 2 at 7 pm"
     - "Show me available restaurants"

3. AI Workflow (LangGraph)
   - Intent Detection: Understands what the user wants (booking, suggestion, browse).
   - Database Search: Retrieves matching restaurants using vector similarity.
   - AI Response: Uses the DeepSeek model through OpenRouter API to generate a structured answer.

4. Output on Streamlit UI
   - Shows the main answer.
   - Also provides extra suggestions and tips in bullet point format.

----------------------------------------------------
How to Run
----------
1. Install Requirements
   pip install fastapi uvicorn pandas chromadb requests streamlit langgraph

2. Update API Key
   - In the main script, replace:
     OPENROUTER_API_KEY = "your_api_key_here"
   - with your valid OpenRouter API Key.

3. Run the Application
   python main.py

4. Access the App
   - FastAPI backend will run at: http://127.0.0.1:8000
   - Streamlit frontend will open automatically in the browser.

----------------------------------------------------
How to Use
----------
- Upload Data: Go to "Upload restaurant data" section in Streamlit UI and upload .csv/.txt/.do files.
- Query the Assistant: Type your request in the text area (e.g., "Find vegetarian restaurants for dinner").
- Get AI Response: The assistant will show:
  - Main Answer (direct result based on your query and data).
  - Suggestions (3-5 helpful recommendations or tips).

----------------------------------------------------
Benefits of This Project
------------------------
- End-to-End Demo: Combines backend (FastAPI) + frontend (Streamlit) + AI + VectorDB in one application.
- Smart Query Understanding: Uses LangGraph to analyze user intent.
- Semantic Search: Finds relevant restaurants using ChromaDB.
- AI-Powered Recommendations: Provides intelligent and user-friendly suggestions.
- Multiple File Formats Supported: Works with CSV, TXT, DO restaurant datasets.
- Scalable: Can be extended to real booking systems or APIs.
- User-Friendly Interface: Streamlit provides a simple, clean UI.

----------------------------------------------------
Example Queries
---------------
- "Book a table for two at 8 PM"
- "Suggest good Italian restaurants"
- "Show available restaurants nearby"

----------------------------------------------------
Important Notes
---------------
- This demo does not actually book tables, it only suggests options.
- Requires internet connection for AI (OpenRouter API).
- Ensure port 8000 is free before running.

====================================================
Developed as a Demo Project using:
Python | FastAPI | Streamlit | LangGraph | ChromaDB | OpenRouter AI
====================================================
