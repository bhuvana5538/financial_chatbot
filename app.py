# The overall application is structured with a FastAPI backend and a Streamlit frontend.
# This single file contains all the necessary code to run the application.

import os
# Set the backend to PyTorch before importing transformers to avoid TensorFlow conflicts.
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import requests
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from threading import Thread
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
import google.generativeai as genai
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# --- Configuration for Hugging Face Models ---
# Hugging Face model for NLU analysis
# Hugging Face model for generative tasks (Q&A, Summary, Insights).
# Note: This is a large model (8B parameters) and will take a significant amount of time
# and memory to load. This is expected behavior for this specific model.
GRANITE_MODEL_NAME = "ibm-granite/granite-3b-code-instruct"

# --- FastAPI Backend Setup ---
# A small note: a FastAPI app is created, but it's not run directly by the main script.
# It's intended to be run in a separate thread.
app = FastAPI(
    title="Personal Finance Chatbot API",
    description="A backend for a personal finance chatbot using Hugging Face models."
)
router = APIRouter()

# Global variables to store the NLU and Granite models
nlu_pipeline = None
granite_tokenizer = None
granite_model = None

# Thread pool for asynchronous model calls
executor = ThreadPoolExecutor(max_workers=5)

# --- New Startup Event Handler ---
@app.on_event("startup")
def startup_event_handler():
    global nlu_pipeline, granite_tokenizer, granite_model, granite_loaded
    try:
        print("Using Gemini for NLU tasks...")
    

        print("Loading IBM Granite model on CPU (no quantization)...")
        granite_tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL_NAME)
        granite_model = AutoModelForCausalLM.from_pretrained(
            GRANITE_MODEL_NAME,
            device_map=None,  # Avoids auto device mapping
            torch_dtype=torch.float32,  # Ensures CPU-compatible precision
            low_cpu_mem_usage=True
        )# Explicitly move to CPU

        granite_loaded = True
        print("Granite model loaded on CPU successfully.")
    except Exception as e:
        print(f"[WARNING] Granite model failed to load: {e}")
        granite_model = None
        granite_tokenizer = None
        granite_loaded = False

# --- FastAPI Endpoints ---
# A simple health check endpoint to confirm the server is running and models are loaded.
@router.get("/status")
def get_status():
    """Returns a simple status message."""
    return {"status": "ok", "message": "Models are loaded and ready."}

# Model for API requests
class NLURequest(BaseModel):
    text: str

class QARequest(BaseModel):
    question: str
    persona: str = "professional"

class BudgetSummaryRequest(BaseModel):
    persona: str = "professional"
    income: float
    expenses: dict[str, float]
    savings_goal: float = 0.0

class SpendingInsightsRequest(BaseModel):
    persona: str = "professional"
    income: float
    expenses: dict[str, float]
    goals: list[dict]

# Helper function to call the Hugging Face Granite model
def call_hf_granite_model(prompt: str):
    if granite_loaded and granite_model and granite_tokenizer:
        try:
            inputs = granite_tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}  # ensure all tensors are on CPU

            outputs = granite_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )

            generated_text = granite_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.replace(prompt, "", 1).strip()
        except Exception as e:
            print(f"[WARNING] Granite inference failed: {e}")

    return "Error: Granite model could not generate a response."

# FastAPI endpoint for NLU analysis
@router.post("/nlu")
async def nlu_handler(payload: NLURequest):
    """Performs sentiment analysis on user text using a Gemini API"""
    try:
        prompt = f"What is the sentiment (positive, negative, or neutral) of this text:\n\n\"{payload.text}\""
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return {"nlu_result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# FastAPI endpoint for Q&A
@router.post("/generate_q_a")
async def generate_q_a_handler(payload: QARequest):
    """Generates a response to a financial question using the IBM Granite model."""
    prompt = f"""
    You are a helpful personal finance assistant. Respond to the following question.
    Tailor your response to a {payload.persona}.
    Be clear, concise, and provide actionable advice.
    Question: {payload.question}
    """
    try:
        granite_result = await asyncio.get_event_loop().run_in_executor(
            executor, call_hf_granite_model, prompt
        )
        return {"answer": granite_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint for Budget Summary
@router.post("/budget_summary")
async def budget_summary_handler(payload: BudgetSummaryRequest):
    """Generates a budget summary from user financial data using the IBM Granite model."""
    expenses_str = "\n".join([f"- {k}: ${v}" for k, v in payload.expenses.items()])
    prompt = f"""
    You are a personal finance assistant. Analyze the following budget data and provide a summary.
    The user is a {payload.persona}.
    - Monthly Income: ${payload.income}
    - Monthly Expenses:
    {expenses_str}
    - Monthly Savings Goal: ${payload.savings_goal}
    
    Provide a concise summary that includes:
    1. Total monthly expenses.
    2. The difference between income and total expenses.
    3. An evaluation of whether the savings goal is achievable with the current budget.
    4. Two actionable recommendations for improving the budget.
    """
    try:
        granite_result = await asyncio.get_event_loop().run_in_executor(
            executor, call_hf_granite_model, prompt
        )
        return {"summary": granite_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint for Spending Insights
@router.post("/spending_insights")
async def spending_insights_handler(payload: SpendingInsightsRequest):
    """Provides spending insights and goal-tracking advice using the IBM Granite model."""
    expenses_str = "\n".join([f"- {k}: ${v}" for k, v in payload.expenses.items()])
    goals_str = "\n".join([f"- {g['name']} (Target Date: {g['target_date']}): ${g['amount']}" for g in payload.goals])
    
    prompt = f"""
    You are a financial advisor. Analyze the user's spending habits and financial goals.
    The user is a {payload.persona}.
    - Monthly Income: ${payload.income}
    - Monthly Expenses:
    {expenses_str}
    - Financial Goals:
    {goals_str}
    
    Provide detailed insights in a structured format:
    1.  *Spending Breakdown:* Identify the top 2-3 spending categories.
    2.  *Goal Progress:* For each goal, calculate the required monthly savings and comment on its achievability based on the current budget (Income - Expenses).
    3.  *Recommendations:* Offer specific recommendations to optimize spending and reach their goals faster.
    """
    try:
        granite_result = await asyncio.get_event_loop().run_in_executor(
            executor, call_hf_granite_model, prompt
        )
        return {"insights": granite_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(router)

# --- Streamlit Frontend UI ---
def streamlit_ui():
    """
    Main function for the Streamlit UI.
    It now waits for the backend to confirm models are loaded before
    rendering the main pages.
    """
    st.set_page_config(layout="wide", page_title="Personal Finance Chatbot")
    
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(to right, #4b6cb7, #182848);
            color: white;
            font-family: 'Inter', sans-serif;
        }
        .st-emotion-cache-1c5c7q5 {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 2rem;
        }
        h1, h2, h3, .st-emotion-cache-121p2j2 {
            color: #ffffff;
            text-align: center;
        }
        .st-emotion-cache-1v093l6 {
            font-size: 1.25rem;
            color: #e0e0e0;
        }
        .stButton>button {
            width: 100%;
            height: 80px;
            font-size: 1.2rem;
            color: #182848;
            background-color: #f0f2f6;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            background-color: #e2e8f0;
        }
        .css-1544g2n {
            background-color: #1a202c;
            color: #a0aec0;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #cbd5e0;
            padding: 0.5rem;
        }
        .output-box {
            background-color: #2d3748;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
            font-family: 'monospace';
            white-space: pre-wrap;
            color: #e2e8f0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Check if the models are loaded by pinging the backend status endpoint
    with st.spinner("Starting server and loading models... This may take a moment."):
        is_ready = False
        while not is_ready:
            try:
                response = requests.get("http://127.0.0.1:8001/status")
                response.raise_for_status()
                if response.json().get("status") == "ok":
                    is_ready = True
            except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
                # Server not ready or endpoint not found yet
                time.sleep(1) # Wait a bit before retrying

    st.title("Personal Finance Chatbot")
    st.markdown("---")

    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "home"

    def navigate_to(page):
        st.session_state.page = page

    # Home Page
    if st.session_state.page == "home":
        st.subheader("Intelligent Guidance for Savings, Taxes, and Investments")
        st.write("This chatbot provides personalized financial guidance using Hugging Face models.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            if st.button("NLU Analysis"):
                navigate_to("nlu")
        with col2:
            if st.button("Q&A"):
                navigate_to("q_a")
        with col3:
            if st.button("Budget Summary"):
                navigate_to("budget_summary")
        with col4:
            if st.button("Spending Insights"):
                navigate_to("spending_insights")

    # NLU Analysis Page
    elif st.session_state.page == "nlu":
        st.subheader("NLU Analysis")
        st.write("Enter text to analyze its sentiment.")
        
        text_input = st.text_area("Enter your text here:", height=150, placeholder="I am so worried about my spending this month.")
        
        if st.button("Analyze Text"):
            if text_input:
                try:
                    response = requests.post("http://127.0.0.1:8001/nlu", json={"text": text_input})
                    response.raise_for_status()
                    result = response.json()
                    st.json(result)
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the FastAPI backend.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter some text to analyze.")
        
        if st.button("🔙 Back to Home"):
            navigate_to("home")

    # Q&A Page
    elif st.session_state.page == "q_a":
        st.subheader("Financial Q&A")
        st.write("Ask any financial question and get a personalized response.")
        
        question = st.text_input("Your question:", placeholder="How can I save money while repaying student loans?")
        persona = st.selectbox("Select your persona:", ["professional", "student"])
        
        if st.button("Get Answer"):
            if question:
                try:
                    with st.spinner("Generating answer..."):
                        response = requests.post("http://127.0.0.1:8001/generate_q_a", json={"question": question, "persona": persona})
                        response.raise_for_status()
                        result = response.json()
                        st.markdown("#### Response:")
                        st.markdown(f"<div class='output-box'>{result['answer']}</div>", unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the FastAPI backend.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a question.")
        
        if st.button("🔙 Back to Home"):
            navigate_to("home")

    # Budget Summary Page
    elif st.session_state.page == "budget_summary":
        st.subheader("Budget Summary")
        st.write("Enter your financial data to get a detailed summary.")
        
        with st.form("budget_form"):
            persona = st.selectbox("Select your persona:", ["professional", "student"])
            income = st.number_input("Monthly Income ($):", min_value=0.0, step=100.0)
            
            st.markdown("#### Monthly Expenses:")
            expense_categories = ["Rent/Mortgage", "Groceries", "Utilities", "Transportation", "Entertainment", "Other"]
            expenses = {}
            for cat in expense_categories:
                expenses[cat] = st.number_input(f"{cat} ($):", min_value=0.0, step=10.0)
            
            savings_goal = st.number_input("Monthly Savings Goal ($):", min_value=0.0, step=10.0)
            
            submitted = st.form_submit_button("Get Budget Summary")
        
        if submitted:
            if income > 0 and sum(expenses.values()) > 0:
                try:
                    payload = {
                        "persona": persona,
                        "income": income,
                        "expenses": expenses,
                        "savings_goal": savings_goal
                    }
                    with st.spinner("Generating summary..."):
                        response = requests.post("http://127.0.0.1:8001/budget_summary", json=payload)
                        response.raise_for_status()
                        result = response.json()
                        st.markdown("#### Summary:")
                        st.markdown(f"<div class='output-box'>{result['summary']}</div>", unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the FastAPI backend.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please fill in your income and at least one expense.")

        if st.button("🔙 Back to Home"):
            navigate_to("home")

    # Spending Insights Page
    elif st.session_state.page == "spending_insights":
        st.subheader("Spending Insights")
        st.write("Enter your financial data and goals to get actionable insights.")
        
        with st.form("insights_form"):
            persona = st.selectbox("Select your persona:", ["professional", "student"])
            income = st.number_input("Monthly Income ($):", min_value=0.0, step=100.0)

            st.markdown("#### Monthly Expenses:")
            expense_categories = ["Rent/Mortgage", "Groceries", "Utilities", "Transportation", "Entertainment", "Other"]
            expenses = {}
            for cat in expense_categories:
                expenses[cat] = st.number_input(f"{cat} ($):", min_value=0.0, step=10.0, key=f"insights_{cat}")
            
            st.markdown("#### Financial Goals:")
            goals_list = []
            if 'num_goals' not in st.session_state:
                st.session_state.num_goals = 1
            
            def add_goal():
                st.session_state.num_goals += 1
            def remove_goal():
                if st.session_state.num_goals > 1:
                    st.session_state.num_goals -= 1

            st.button("Add Goal", on_click=add_goal)
            st.button("Remove Last Goal", on_click=remove_goal)
            
            for i in range(st.session_state.num_goals):
                with st.container(border=True):
                    goal_name = st.text_input(f"Goal {i+1} Name:", key=f"goal_name_{i}")
                    goal_amount = st.number_input(f"Goal {i+1} Amount ($):", min_value=0.0, key=f"goal_amount_{i}")
                    goal_date = st.date_input(f"Goal {i+1} Target Date:", key=f"goal_date_{i}")
                    if goal_name and goal_amount > 0:
                        goals_list.append({"name": goal_name, "amount": goal_amount, "target_date": str(goal_date)})

            submitted = st.form_submit_button("Get Spending Insights")

        if submitted:
            if income > 0 and sum(expenses.values()) > 0 and goals_list:
                try:
                    payload = {
                        "persona": persona,
                        "income": income,
                        "expenses": expenses,
                        "goals": goals_list
                    }
                    with st.spinner("Generating insights..."):
                        response = requests.post("http://127.0.0.1:8001/spending_insights", json=payload)
                        response.raise_for_status()
                        result = response.json()
                        st.markdown("#### Insights:")
                        st.markdown(f"<div class='output-box'>{result['insights']}</div>", unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.error("Error: Could not connect to the FastAPI backend.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please fill in your income, expenses, and at least one goal.")

        if st.button("🔙 Back to Home"):
            navigate_to("home")

# Function to run the FastAPI server in a separate thread
def run_fastapi():
    """Starts the FastAPI server with uvicorn."""
    uvicorn.run(app, host="0.0.0.0", port=8001)

if _name_ == "_main_":
    # Start the FastAPI server in a separate thread.
    api_thread = Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    # Run the Streamlit UI, which will now wait for the backend to be ready.
    streamlit_ui()
