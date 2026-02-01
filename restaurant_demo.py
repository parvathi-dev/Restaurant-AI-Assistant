import os
import json
import tempfile
import threading
import time

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import chromadb
import requests
import streamlit as st

from langgraph.graph import StateGraph, END

# CONFIG - Configuration settings for the application
# API key for OpenRouter AI service
OPENROUTER_API_KEY ="Your_API_Key_Here"

API_HOST = '127.0.0.1'  # Local host address
API_PORT = 8000  # Port for FastAPI backend
API_URL = f"http://{API_HOST}:{API_PORT}"  # Complete API URL

############## FASTAPI + VECTOR DB ############
# This function creates and runs the FastAPI backend server


def start_fastapi():
    # Create FastAPI application instance
    app = FastAPI()

    # Add CORS middleware to allow cross-origin requests (needed for Streamlit to communicate with FastAPI)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow requests from any origin
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],  # Allow all headers
    )

    # Initialize ChromaDB client - this is our vector database for storing restaurant data
    chroma_client = chromadb.Client()
    # Create or get existing collection named "restaurants"
    collection = chroma_client.get_or_create_collection("restaurants")

    # Endpoint to upload and process restaurant data files
    @app.post("/upload/")
    async def upload(file: UploadFile = File(...)):
        # Get file extension to determine how to parse it
        ext = file.filename.split('.')[-1]

        # Create temporary file to save uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(await file.read())
            fname = tmp.name

        try:
            # Parse different file types
            if ext == 'csv':
                df = pd.read_csv(fname)  # Standard CSV parsing
            elif ext in ['txt', 'do']:
                df = pd.read_csv(fname, delimiter='|')  # Pipe-delimited files
            else:
                return {"error": "Invalid file type"}
        except Exception as e:
            return {"error": f"File parse error: {e}"}

        # Add each row from the dataframe to the vector database
        for idx, row in df.iterrows():
            try:
                # Convert row to JSON string for storage
                doc = row.to_json()
                # Add document to ChromaDB collection
                collection.add(
                    documents=[doc],  # The actual data
                    # Metadata for searching
                    metadatas=[{"name": str(row.get("name", f"row_{idx}"))}],
                    # Unique ID for each document
                    ids=[f"{idx}_{os.urandom(4).hex()}"]
                )
            except Exception:
                # Skip rows that cause errors
                pass

        # Clean up temporary file
        os.remove(fname)
        return {"success": True, "rows": len(df)}

    # Endpoint to retrieve all restaurants from the database
    @app.get("/restaurants/")
    async def get_restaurants():
        try:
            # Get all documents with their metadata from the collection
            docs = collection.get(include=["documents", "metadatas"])
            all_docs = collection.get()

            return {
                "documents": docs["documents"] if docs and "documents" in docs else [],
                "metadatas": docs["metadatas"] if docs and "metadatas" in docs else [],
                "ids": all_docs["ids"] if all_docs and "ids" in all_docs else []
            }
        except Exception as e:
            # Return empty results if there's an error
            return {
                "documents": [],
                "metadatas": [],
                "ids": [],
                "error": str(e)
            }

    # Main endpoint to process user queries using LangGraph
    @app.post("/process/")
    async def process(prompt: str = Form(...)):

        # NODE 1: Determine user's intent from their input
        def determine_intent(state):
            """
            Analyzes user input to classify what they want to do.
            LangGraph State: A dictionary that gets passed between nodes
            """
            text = state["user_input"].strip().lower()

            # Check for booking-related keywords
            if any(x in text for x in ["book", "reserve", "table"]):
                state["intent"] = "booking"
            # Check for suggestion/search keywords
            elif any(x in text for x in ["suggest", "recommend", "find", "search"]):
                state["intent"] = "suggest"
            # Check for browsing keywords
            elif any(x in text for x in ["show", "list", "available"]):
                state["intent"] = "browse"
            else:
                state["intent"] = "other"

            return state  # Always return the state in LangGraph nodes

        # NODE 2: Search the vector database for relevant restaurants
        def search_db(state):
            """
            Queries ChromaDB to find restaurants relevant to user's request
            """
            q = state["user_input"]
            try:
                # Use ChromaDB's semantic search to find similar documents
                results = collection.query(
                    query_texts=[q],  # User's query
                    n_results=5,      # Return top 5 matches
                    # Include both content and metadata
                    include=['documents', 'metadatas']
                )

                # Extract documents from results
                docs = results['documents'][0] if results['documents'] else []
                state["db_results"] = docs
                state["context"] = {"db_results": docs}
            except Exception as e:
                # Handle database errors gracefully
                state["db_results"] = []
                state["context"] = {"db_results": [], "error": str(e)}

            return state

        # NODE 3: Generate AI response using the LLM
        def run_llm(state):
            """
            Calls the AI model to generate a response based on restaurant data and user intent
            """
            # Prepare context from database results
            context_lines = []
            for doc in state.get("db_results", []):
                try:
                    # Parse JSON document and format it nicely
                    j = json.loads(doc)
                    context_lines.append(
                        ", ".join(f"{k}: {v}" for k, v in j.items()))
                except Exception:
                    # If JSON parsing fails, use raw document
                    context_lines.append(str(doc))

            # Limit context to prevent token overflow (2000 chars max)
            joined_context = "\n".join(context_lines)[:2000]

            # Create different prompts based on user intent
            if state["intent"] == "booking":
                user_prompt = (
                    f"You are a helpful restaurant assistant. The user wants to book a table.\n"
                    f"Based on the restaurant data below, help them understand booking options but DO NOT actually book.\n"
                    f"Instead, tell them which restaurants they can book and provide booking suggestions.\n\n"
                    f"Restaurant data:\n{joined_context}\n\n"
                    f"User request:\n{state['user_input']}\n\n"
                    f"Format your response as:\n"
                    f"1. Main response about booking possibilities\n"
                    f"2. At the end, add 'Suggestions:' and list 3-5 helpful tips or alternatives, each on a new line with '• '"
                )
            else:
                user_prompt = (
                    f"You are a helpful restaurant assistant.\n"
                    f"Use the following restaurant data to help the user.\n\n"
                    f"Restaurant data:\n{joined_context}\n\n"
                    f"User request:\n{state['user_input']}\n\n"
                    f"Format your response as:\n"
                    f"1. Main response to their query\n"
                    f"2. At the end, add 'Suggestions:' and list 3-5 helpful tips or recommendations, each on a new line with '• '"
                )

            # Prepare API call to OpenRouter
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "x-ai/grok-4.1-fast",  
                "messages": [{"role": "user", "content": user_prompt}]
            }

            try:
                # Make API call to get AI response
                response = requests.post(
                    url, headers=headers, data=json.dumps(data), timeout=30)
                if response.status_code == 200:
                    r = response.json()
                    state["answer"] = r["choices"][0]["message"]["content"]
                else:
                    state["answer"] = f"Error {response.status_code}: {response.text}"
            except Exception as e:
                state["answer"] = f"Exception calling LLM: {e}"

            return state

        # CONDITIONAL EDGE: Decides which node to go to next based on intent
        def intent_router(state):
            """
            Routes the workflow based on user intent
            This is a conditional edge in LangGraph
            """
            if state["intent"] in ("browse", "suggest", "booking"):
                return "search_db"  # These intents need database search
            return "llm"  # Other intents can go directly to LLM

        # BUILD THE LANGGRAPH WORKFLOW
        # StateGraph manages the flow between different processing steps
        builder = StateGraph(dict)  # State is a dictionary

        # Add nodes (processing steps) to the graph
        builder.add_node("intent", determine_intent)  # Step 1: Analyze intent
        builder.add_node("search_db", search_db)      # Step 2: Search database
        # Step 3: Generate response
        builder.add_node("llm", run_llm)

        # Define the workflow flow
        builder.set_entry_point("intent")  # Start with intent analysis

        # Add conditional routing based on intent
        builder.add_conditional_edges(
            "intent",           # From intent node
            intent_router,      # Use this function to decide
            {
                "search_db": "search_db",  # If router returns "search_db", go to search_db node
                "llm": "llm"               # If router returns "llm", go to llm node
            }
        )

        # Add edges (connections between nodes)
        # After searching, always go to LLM
        builder.add_edge("search_db", "llm")
        builder.add_edge("llm", END)          # After LLM, end the workflow

        # Compile the graph into an executable workflow
        graph = builder.compile()

        # EXECUTE THE WORKFLOW
        # Initialize the state with user input
        state = {
            "user_input": prompt,
            "intent": "",
            "context": {},
            "db_results": [],
            "answer": ""
        }

        # Run the workflow and get final state
        final = graph.invoke(state)
        return {"response": final.get("answer", "")}

    # Start the FastAPI server
    uvicorn.run(app, host=API_HOST, port=API_PORT)

############## STREAMLIT UI ##############
# This function creates the web interface using Streamlit


def start_streamlit():
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Restaurant App (LangGraph AI)", layout="centered")
    st.title("🍴 Restaurant Demo (LangGraph-powered)")
    st.info("Upload restaurant data. Chat, book, consult, get suggestions using AI + vector DB.")

    # FILE UPLOAD SECTION
    # Expandable section for uploading restaurant data
    with st.expander("Upload restaurant data (.csv/.txt/.do)"):
        uploaded = st.file_uploader("Choose file", type=["csv", "txt", "do"])

        if uploaded is not None:
            # Prepare file for upload to FastAPI backend
            files = {'file': (uploaded.name, uploaded.read(), uploaded.type)}

            # Show loading spinner while uploading
            with st.spinner("Uploading and processing file..."):
                resp = requests.post(f"{API_URL}/upload/", files=files)

            # Handle upload response
            if resp.status_code == 200 and 'rows' in resp.json():
                st.success(f"Uploaded! Rows: {resp.json().get('rows', 0)}")
            elif resp.status_code == 200:
                st.error(f"Upload error: {resp.json()}")
            else:
                st.error(f"Upload failed: {resp.text}")

    st.divider()  # Visual separator

    # AI ASSISTANT SECTION
    st.subheader("🤖 Restaurant AI Assistant")
    st.markdown(
        "Ask me to help you find restaurants, get suggestions, or check booking options!")

    # Text area for user input
    prompt = st.text_area(
        "Your request:",
        placeholder="e.g., 'I want to book a table for 2 at 7pm' or 'Suggest good Italian restaurants' or 'Find restaurants open now'",
        height=100
    )

    # Main interaction button
    if st.button("🚀 Ask AI Assistant", type="primary"):
        if not prompt.strip():
            st.warning("Please enter your request.")
        else:
            try:
                # Show loading spinner while AI processes the request
                with st.spinner("🤔 AI is thinking... Please wait"):
                    # Send request to FastAPI backend
                    resp = requests.post(
                        f"{API_URL}/process/", data={"prompt": prompt})

                if resp.status_code == 200:
                    response_text = resp.json().get("response", "No answer from AI.")

                    # Parse and display response with nice formatting
                    if "Suggestions:" in response_text:
                        # Split main response from suggestions
                        main_response, suggestions = response_text.split(
                            "Suggestions:", 1)

                        # Display main response
                        st.success("✅ Response:")
                        st.write(main_response.strip())

                        # Display suggestions in a formatted way
                        st.info("💡 Suggestions:")
                        suggestion_lines = [
                            line.strip() for line in suggestions.split('\n') if line.strip()]
                        for suggestion in suggestion_lines:
                            if suggestion.startswith('•'):
                                st.write(suggestion)
                            elif suggestion:
                                st.write(f"• {suggestion}")
                    else:
                        # Display simple response if no suggestions format
                        st.success("✅ Response:")
                        st.write(response_text)

                else:
                    st.error(f"❌ API error: {resp.status_code}")

            except requests.exceptions.RequestException:
                st.error(
                    "❌ Backend not started or connection issue. Please make sure the server is running.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

############# MULTI-THREAD MAIN #############
# Main function that starts both FastAPI and Streamlit


def main():
    """
    This function runs both the backend (FastAPI) and frontend (Streamlit) simultaneously
    using threading so they can run concurrently
    """
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()

    # Wait a bit for the API to start up
    time.sleep(3)

    # Start Streamlit (this blocks, so it runs last)
    start_streamlit()


# Entry point - runs when script is executed directly
if __name__ == "__main__":
    main()
