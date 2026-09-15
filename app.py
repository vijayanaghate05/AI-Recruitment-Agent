import os

import streamlit as st
from dotenv import load_dotenv

from agents import ResumeAnalysisAgent
from ui import render_sidebar, render_dashboard

load_dotenv()

ROLE_REQUIREMENTS = {
    "AI/ML Engineer": ["Python", "PyTorch", "TensorFlow", "LangChain", "RAG", "LLMs", "FAISS"],
    "Data Scientist": ["Python", "Pandas", "Scikit-Learn", "SQL", "Statistics", "Data Visualization"],
    "Software Engineer": ["Data Structures", "Algorithms", "System Design", "Git", "CI/CD"],
    "Backend Engineer": ["Python", "FastAPI", "PostgreSQL", "Docker", "REST APIs", "Redis"],
    "Frontend Engineer": ["React", "TypeScript", "JavaScript", "HTML/CSS", "State Management"],
    "Data Engineer": ["Python", "Spark", "SQL", "ETL", "Airflow", "Data Warehousing"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Linux"],
}

st.set_page_config(
    page_title="AI Recruitment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_agent(api_key: str) -> ResumeAnalysisAgent:
    """Keep the agent in this browser session instead of recreating it on every rerun."""
    saved_key = st.session_state.get("agent_api_key")
    if "agent" not in st.session_state or saved_key != api_key:
        st.session_state.agent = ResumeAnalysisAgent(
            api_key=api_key,
            role_requirements=ROLE_REQUIREMENTS,
        )
        st.session_state.agent_api_key = api_key
        st.session_state.resume_processed = False
        st.session_state.analysis_result = None
        st.session_state.generated_questions = []
        st.session_state.generated_suggestions = []

    return st.session_state.agent


api_key = render_sidebar()

if not api_key:
    st.info("Enter your Groq API key in the sidebar, or set GROQ_API_KEY in your .env file.")
else:
    try:
        agent = get_agent(api_key)
        render_dashboard(agent, ROLE_REQUIREMENTS)
    except Exception as exc:
        st.error("The application could not start.")
        st.exception(exc)
