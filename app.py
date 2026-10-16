# app.py
# ================================
# IMPORTS
# ================================
import os
import json
import faiss
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================================
# STREAMLIT UI CONFIG
# ================================
st.set_page_config(
    page_title="InsightAI | Modern Personality Assessment", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Modern, Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }

    /* Main Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        padding: 45px 30px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.25);
        margin-bottom: 35px;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
    }

    /* Card Styling */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }

    /* Question Box Styling */
    .q-number {
        font-size: 0.85rem;
        font-weight: 700;
        color: #c084fc;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }

    .q-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    /* Team Cards */
    .team-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
        border-color: rgba(168, 85, 247, 0.5);
    }

    .team-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 10px;
    }

    .team-role {
        font-size: 0.9rem;
        color: #c084fc;
        font-weight: 600;
    }

    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        border: none !important;
        box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4) !important;
        width: 100%;
        font-size: 1.05rem !important;
    }

    .stButton > button:hover {
        opacity: 0.95;
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# CONFIGURE LLM (GROQ API)
# ================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You generate structured JSON personality questions only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return completion.choices[0].message.content.strip()

def generate_personality_questions():
    prompt = """
    Acting as an expert psychologist. You are creating a personality assessment based on the Big Five traits:
- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

REQUIREMENTS:
- Generate EXACTLY 15 questions personality assessment based on the TIPI Big Five model.
- Each question must be a natural full sentence.
- Questions may implicitly measure one or multiple traits, but DO NOT mention the traits in the output.
- Questions should feel conversational, real, and not scientific.
- Output JSON ONLY as:
JSON FORMAT:
{
  "questions": [
     "Question 1",
     ...
     "Question 15"
  ],
  "trait_mapping": [
     {"index": 0, "traits": ["Extraversion"]},
     ...
  ]
}

Allowed trait names:
Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness

No extra text, no comments, no formatting outside JSON. No explanations. No markdown.
    """

    output = call_llm(prompt)
    output = output.replace("```json","").replace("```","").strip()

    try:
        data = json.loads(output)
        final = []
        for i, qtext in enumerate(data["questions"]):
            trait = "Openness"
            for tm in data.get("trait_mapping", []):
                if tm.get("index") == i and tm.get("traits"):
                    trait = tm["traits"][0]
                    break

            final.append({
                "question": qtext,
                "options": [
                    "Strongly Disagree",
                    "Disagree",
                    "Neutral",
                    "Agree",
                    "Strongly Agree"
                ],
                "trait": trait
            })
        return final
    except Exception as e:
        return [{"error": "INVALID JSON RETURNED"}]

def score_answers(answers):
    trait_scores = {}
    trait_counts = {}
    for ans in answers:
        trait = ans["trait"]
        value = ans["value"]
        if trait not in trait_scores:
            trait_scores[trait] = 0
            trait_counts[trait] = 0
        trait_scores[trait] += value
        trait_counts[trait] += 1

    for t in trait_scores:
        trait_scores[t] = round((trait_scores[t] / (trait_counts[t] * 5)) * 100)
    return trait_scores

def generate_report(trait_scores):
    prompt = f"""
You are a personality psychology expert.
Based on these Big Five scores: {trait_scores}

Write a personalized report in second person.
1. A 3-4 sentence personality summary.
2. 4 strengths (bullet list)
3. 4 growth areas (bullet list)
4. 4 actionable recommendations

Use friendly, conversational language. No headings, just formatted text.
"""
    return call_llm(prompt)

# FAISS Vector Store Setup
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(384)
    st.session_state.documents = []

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def store_in_faiss(text):
    vec = embed_model.encode([text])
    st.session_state.faiss_index.add(np.array(vec).astype('float32'))
    st.session_state.documents.append(text)

def rag_chat(user_query):
    if len(st.session_state.documents) == 0:
        return "No personality profile found yet. Please complete the test on the Home tab first!"

    query_vec = embed_model.encode([user_query])
    distances, neighbors = st.session_state.faiss_index.search(np.array(query_vec).astype('float32'), 3)
    neighbor_idxs = [int(i) for i in neighbors[0] if i != -1]
    retrieved = "\n\n".join(st.session_state.documents[i] for i in neighbor_idxs)

    prompt = f"""
Based only on this personality profile:
{retrieved}

Answer the user's question in plain text. Always address the user as 'you/your'. Make it friendly and insightful.

QUESTION: {user_query}
"""
    return call_llm(prompt)

# ================================
# HEADER & NAVIGATION
# ================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 InsightAI Personality Engine</div>
    <div class="hero-subtitle">Discover deep psychological insights & personalized AI coaching using the Big Five Model</div>
</div>
""", unsafe_allow_html=True)

if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "scores" not in st.session_state:
    st.session_state.scores = {}
if "report" not in st.session_state:
    st.session_state.report = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

home_tab, chat_tab, dashboard_tab, about_tab = st.tabs(["✨ Take Assessment", "💬 AI Coach Chat", "📊 Visual Dashboard", "👥 About Us"])

# ==========================
# PAGE 1: HOME ASSESSMENT
# ==========================
with home_tab:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Welcome to your Assessment")
        st.write("Click below to generate an AI-powered personalized psychological assessment.")
    with col2:
        if st.button("🚀 Start Assessment"):
            st.session_state.questions = generate_personality_questions()
            st.session_state.answers = [None] * len(st.session_state.questions)
            st.session_state.show_test = True

    if st.session_state.get("show_test", False) and st.session_state.questions:
        st.markdown("---")
        for i, q in enumerate(st.session_state.questions):
            st.markdown(f"""
            <div class="custom-card">
                <div class="q-number">Question {i+1} of {len(st.session_state.questions)}</div>
                <div class="q-text">{q['question']}</div>
            </div>
            """, unsafe_allow_html=True)

            selected = st.radio(
                label="",
                options=q["options"],
                key=f"q{i}",
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state.answers[i] = selected

        if st.button("🎯 Submit & Generate Insights"):
            unanswered = [idx for idx, an in enumerate(st.session_state.answers) if an is None or an == ""]
            if unanswered:
                st.error(f"Please answer all questions before submitting. Unanswered: {', '.join(str(i+1) for i in unanswered)}")
            else:
                mapping = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5}
                collected = []
                for ans_value, q in zip(st.session_state.answers, st.session_state.questions):
                    collected.append({
                        "trait": q.get("trait", "Openness"),
                        "value": mapping.get(ans_value, 3)
                    })

                st.session_state.scores = score_answers(collected)
                st.session_state.report = generate_report(st.session_state.scores)
                store_in_faiss(st.session_state.report)
                st.success("🎉 Assessment Complete! Go to the 'Visual Dashboard' tab to view your full report.")

# ==========================
# PAGE 2: CHATBOT
# ==========================
with chat_tab:
    st.subheader("💬 AI Personality Coach & Advisor")
    st.caption("Ask questions about your strengths, career suitability, or growth strategies based on your results.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["message"])

    user_input = st.chat_input("Ask your coach anything...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        response = rag_chat(user_input)
        st.session_state.chat_history.append({"role": "assistant", "message": response})
        with st.chat_message("assistant"):
            st.write(response)

# ==========================
# PAGE 3: DASHBOARD
# ==========================
with dashboard_tab:
    if not st.session_state.get('scores'):
        st.info("👈 Please complete the assessment test first to unlock your visual analytics dashboard.")
    else:
        st.subheader("📊 Psychological Profile Analytics")

        traits = list(st.session_state.scores.keys())
        scores = list(st.session_state.scores.values())

        c1, c2 = st.columns(2)

        with c1:
            fig_bar = go.Figure([go.Bar(
                x=traits, 
                y=scores, 
                marker_color=['#818cf8', '#c084fc', '#f472b6', '#38bdf8', '#4ade80']
            )])
            fig_bar.update_layout(
                title="Big Five Trait Metrics (%)", 
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_pie = go.Figure(data=[go.Pie(labels=traits, values=scores, hole=0.4)])
            fig_pie.update_layout(
                title="Personality Balance Index", 
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("""
        <div class="custom-card">
            <h3 style="color:#c084fc;">📄 Detailed Personality Insights Report</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.report.replace("\n", "  \n"))

# ==========================
# PAGE 4: ABOUT US
# ==========================
with about_tab:
    st.subheader("👥 Project Development Team")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="team-card">
            <div style="font-size: 3rem;">👩‍💻</div>
            <div class="team-name">Laiba Ishfaq</div>
            <div class="team-role">Lead AI Developer</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
        <div class="team-card">
            <div style="font-size: 3rem;">👩‍💻</div>
            <div class="team-name">Farah Maktaqir</div>
            <div class="team-role">AI Developer & Research Partner</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="custom-card">
        <h3>About InsightAI System</h3>
        <p style="color:#cbd5e1; line-height:1.6;">
            The AI Personality Assessment & Insight System evaluates Big Five psychological dimensions using state-of-the-art LLMs (Llama 3.3 via Groq) and RAG vector search (FAISS) to provide personalized growth actionable insights, career guidance, and interactive AI coaching.
        </p>
    </div>
    """, unsafe_allow_html=True)