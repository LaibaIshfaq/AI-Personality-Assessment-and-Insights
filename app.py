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

# ================================
# GENERATE TEST QUESTIONS (JSON)
# ================================
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

No extra text, no comments, no formatting outside JSON.
    No explanations. No markdown.
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
        print("LLM OUTPUT:", output)
        print("JSON parse error:", e)
        return [{"error": "INVALID JSON RETURNED"}]

# ================================
# SCORING THE TEST
# ================================
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

# ================================
# GENERATE LLM PERSONALITY REPORT
# ================================
def generate_report(trait_scores):
    prompt = f"""
You are a personality psychology expert.

Based on these Big Five scores:
{trait_scores}

Write a **personalized report in second person**.
1. A 3-4 sentence personality summary.
2. 4 strengths (bullet list)
3. 4 growth areas (bullet list)
4. 4 actionable recommendations

Use friendly, conversational language.
No headings, just formatted text.
"""
    return call_llm(prompt)

#==========================
# Save FAISS ACROSS SESSION
#==========================
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(384)
    st.session_state.documents = []

# ================================
# SETUP FAISS VECTOR STORE
# ================================
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = 384

def store_in_faiss(text):
    vec = embed_model.encode([text])
    st.session_state.faiss_index.add(np.array(vec).astype('float32'))
    st.session_state.documents.append(text)

# ================================
# RAG CHATBOT
# ================================
def rag_chat(user_query):
    if len(st.session_state.documents) == 0:
        return "No personality profile found yet. Take the test first."

    query_vec = embed_model.encode([user_query])
    distances, neighbors = st.session_state.faiss_index.search(np.array(query_vec).astype('float32'), 3)

    neighbor_idxs = [int(i) for i in neighbors[0] if i != -1]
    retrieved = "\n\n".join(st.session_state.documents[i] for i in neighbor_idxs)

    prompt = f"""
Based only on this personality profile:

{retrieved}

Answer the user's question in plain text, not JSON.
Answer in second person always try to address the user with you, your etc.
Use bullet points if multiple items, and make it friendly and readable.

QUESTION: {user_query}
"""
    answer = call_llm(prompt)
    return answer

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="AI Personality Assessment & Insight System", page_icon="🧠", layout="wide")

st.markdown(
    """
    <div style="background: linear-gradient(90deg, #ff7eb9, #ff758c); padding: 20px; border-radius: 10px">
        <h1 style="color: white; text-align: center;">🧠 AI Personality Assessment & Coach</h1>
    </div>
    """, unsafe_allow_html=True
)

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

tabs = st.tabs(["Home", "Chat", "Dashboard", "About Us"])
home_tab, chat_tab, dashboard_tab, about_tab = tabs

# ==========================
# PAGE 1: HOME
# ==========================
with home_tab:
    st.markdown(
        """
        <div style="
            position: relative;
            text-align: center;
            color: white;
            margin-bottom: 35px;
        ">
             <img src='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUSEhIVFhUXFRUVFRUYFRYVFxcWFhUXFxUXFRcYHSggGBolGxUVITEhJSkrLi4uGB8zODMtNygtLisBCgoKDg0OGxAQGi0mHyYvLS0tLS8tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0vKy0tLS0tLS0tLS0tLy0tLf/AABEIAHIBuwMBIgACEQEDEQH/xAAbAAABBQEBAAAAAAAAAAAAAAADAAIEBQYBB//EAEAQAAIBAgQDBgQDBwMBCQAAAAECEQADBBIhMQVBUQYTImFxgTKRobFCwdEUI1JicuHwgsLxFQczQ1NjkqKy0v/EABoBAAMBAQEBAAAAAAAAAAAAAAECAwQABQb/xAAwEQACAgEDAgQFAwQDAAAAAAAAAQIRAxIhMQRBEyJRYTKhscHwBXGBFJHh8SMz0f/aAAwDAQACEQMRAD8A8PBoimhiuinTFaDTXRQgaeDTWK0PFdpoNdmiIx4ropmalmrgUELUwmmzTgKJ1UICiotcUUZBRSElI6ooqikq0RUqiRCUhyCjLTFWiKKZEZMKtEWhrRBTEmESirQ1FFWmRKQ8U9RTVFFQUyISYVRXTTgKaachY2jYPCtduLbT4mMDoOpPkBJ9qGBWp7J4Egd4N3JA0ki2mrkerZVHmCKSbqLaK4oqU0nwarhnZDJZhdFMF2jxXI1gxynlsPmSFVfO7PltW7SiJYSzMYQELJA3aP5RvVxie0y2/wByJkeCRqQeZHWD9uVZTF2ScalengUzqSlZlBYhY8T1hEwH0hFm407T1YnU6gAabxQ7D1bK3gJqB3O2d4mN+gA1qg4rxkYq53l3O/O45n02GnsAABVOvjF8yY+1c0Y9sMvVvE8c4qXm2d5pPmc60cW8YnN4SfeKtcLxY258Qc5SAdNNjA31mNfWsEjsd60fAuJqj95fF1c7lVmdT06qA361W5w2d8Oef0zfhXjGaxvB+U78/f/AIr0W08tLg4n8+e31+f6004j3j4n8tT+dO1w4sI9P1zV3G49d211qtw/Evd5Vubn15211rN8J7R37AIV9Oup17XJqCj2qPif/tV5f1B48Zc9v61qCnjXy6/vSpVRv0f/9k='
              style='width:100%; border-radius:15px; max-height:400px; object-fit:cover;'/>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Start Personality Test", key="generate_test_home"):
        st.session_state.questions = generate_personality_questions()
        st.session_state.answers = [None] * len(st.session_state.questions)
        st.session_state.show_test = True

    if st.session_state.get("show_test", False) and st.session_state.questions:
        st.markdown(
            "<h4 style='color:#4a00e0'>Please answer every question to get the most accurate personality assessment.</h4>",
            unsafe_allow_html=True
        )

        for i, q in enumerate(st.session_state.questions):
            question_style = "border: 1px solid #ddd; padding:10px; border-radius:10px; margin-bottom:10px;"
            st.markdown(f'<div style="{question_style}">{q["question"]}</div>', unsafe_allow_html=True)

            selected = st.radio(
                label="",
                options=q["options"],
                key=f"q{i}",
                label_visibility="collapsed"
            )
            st.session_state.answers[i] = selected

        if st.button("Submit Test", key="submit_test_home"):
            unanswered = [idx for idx, an in enumerate(st.session_state.answers) if an is None or an == ""]
            if unanswered:
                st.error(f"Please answer all questions. Unanswered: {', '.join(str(i+1) for i in unanswered)}")
            else:
                mapping = {
                    "Strongly Disagree": 1,
                    "Disagree": 2,
                    "Neutral": 3,
                    "Agree": 4,
                    "Strongly Agree": 5
                }

                collected = []
                for ans_value, q in zip(st.session_state.answers, st.session_state.questions):
                    trait = q.get("trait", "Openness")
                    collected.append({
                        "trait": trait,
                        "value": mapping.get(ans_value, 3)
                    })

                st.session_state.scores = score_answers(collected)
                st.session_state.report = generate_report(st.session_state.scores)

                store_in_faiss(st.session_state.report)

                st.success("Test submitted! You can view results on the Dashboard tab.")

# ==========================
# PAGE 2: CHAT
# ==========================
with chat_tab:
    st.subheader("💬 Chat with Personality Coach")

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    def send_message():
        user_msg = st.session_state.chat_input.strip()
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "message": user_msg})
            ai_msg = rag_chat(user_msg)
            st.session_state.chat_history.append({"role": "ai", "message": ai_msg})
        st.session_state.chat_input = ""

    st.markdown("""
        <style>
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
            padding: 10px;
            background: linear-gradient(180deg, #f3e5ff 0%, #e0bbff 100%);
            border-radius: 15px;
        }
        .chat-message {
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-size: 16px;
        }
        .user-message {
            background: linear-gradient(90deg, #8e2de2, #4a00e0);
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .ai-message {
            background: linear-gradient(90deg, #d0b3ff, #a27eff);
            color: #1a1a1a;
            margin-right: auto;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

    chat_html = "<div class='chat-container'>"
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            chat_html += f"<div class='chat-message user-message'>{chat['message']}</div>"
        else:
            chat_html += f"<div class='chat-message ai-message'>{chat['message']}</div>"
    chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)

    st.text_input(
        "Type your message here...",
        key="chat_input",
        on_change=send_message,
    )

# ==========================
# PAGE 3: DASHBOARD
# ==========================
with dashboard_tab:
    if not st.session_state.get('scores'):
        st.warning("Please complete the test first.")
    else:
        st.subheader("📊 Personality Dashboard")

        traits = list(st.session_state.scores.keys())
        scores = list(st.session_state.scores.values())
        bar_colors = ['#ff7eb9','#ff758c','#ff9472','#ffcc70','#70d6ff']

        fig_bar = go.Figure([go.Bar(x=traits, y=scores, marker_color=bar_colors)])
        fig_bar.update_layout(title="Big Five Trait Scores", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = go.Figure(data=[go.Pie(labels=traits, values=scores, hole=0.3)])
        fig_pie.update_traces(marker=dict(colors=bar_colors))
        fig_pie.update_layout(title="Trait Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        df_area = pd.DataFrame({
            'Traits': traits,
            'Score1': [s*0.8 for s in scores],
            'Score2': [s*0.6 for s in scores],
            'Score3': [s*0.4 for s in scores]
        })
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_area['Traits'], y=df_area['Score1'], fill='tozeroy', name='Level 1'))
        fig_area.add_trace(go.Scatter(x=df_area['Traits'], y=df_area['Score2'], fill='tonexty', name='Level 2'))
        fig_area.add_trace(go.Scatter(x=df_area['Traits'], y=df_area['Score3'], fill='tonexty', name='Level 3'))
        fig_area.update_layout(title="Trait Progression Over Levels")
        st.plotly_chart(fig_area, use_container_width=True)

        fig_donut = go.Figure(data=[go.Pie(labels=traits, values=scores, hole=0.5)])
        fig_donut.update_traces(marker=dict(colors=bar_colors))
        fig_donut.update_layout(title="Donut Chart of Traits")
        st.plotly_chart(fig_donut, use_container_width=True)

        df_stacked = pd.DataFrame({
            'Traits': traits,
            'Part1': [s*0.5 for s in scores],
            'Part2': [s*0.3 for s in scores],
            'Part3': [s*0.2 for s in scores]
        })
        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(x=df_stacked['Traits'], y=df_stacked['Part1'], name='Part 1'))
        fig_stacked.add_trace(go.Bar(x=df_stacked['Traits'], y=df_stacked['Part2'], name='Part 2'))
        fig_stacked.add_trace(go.Bar(x=df_stacked['Traits'], y=df_stacked['Part3'], name='Part 3'))
        fig_stacked.update_layout(barmode='stack', title="Stacked Trait Components")
        st.plotly_chart(fig_stacked, use_container_width=True)

        st.markdown("### Summary & Recommendations")
        st.markdown(st.session_state.report.replace("\n", "  \n"))

# ==========================
# PAGE 4: ABOUT US
# ==========================
with about_tab:
    st.subheader("About Us")

    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #7F00FF, #E100FF);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        ">
        Meet Our Team
        </div>
        """,
        unsafe_allow_html=True
    )

    team_members = [
        {"name": "Laiba Ishfaq", "role": "Developer", "image": "https://via.placeholder.com/100"},
        {"name": "Farah Maktaqir", "role": "Developer", "image": "https://via.placeholder.com/100"},
    ]

    cards_html = """
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        margin-top: 20px;
    ">
    """

    for member in team_members:
        cards_html += f"""
        <div style="
            background: linear-gradient(135deg, #B993D6, #8CA6DB);
            border-radius: 12px;
            padding: 15px;
            width: 230px;
            text-align: center;
            color: white;
            box-shadow: 3px 3px 15px rgba(0,0,0,0.2);
        ">
            <img src="{member['image']}" width="100" style="border-radius: 50%; margin-bottom: 10px;">
            <h3 style="margin: 5px; font-size: 18px;">{member['name']}</h3>
            <h4 style="margin: 5px; font-weight: normal; font-size: 15px;">{member['role']}</h4>
        </div>
        """

    cards_html += "</div>"

    import streamlit.components.v1 as components
    components.html(cards_html, height=420)
    
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #A18CD1, #FBC2EB);
            padding: 20px;
            border-radius: 12px;
            margin-top: 30px;
            color: #1A1A1A;
        ">
            <h2 style="text-align:center;">About the Project</h2>
            <p style="font-size:16px; line-height:1.6;">
                The AI Personality Assessment & Insight System is a cutting-edge application that evaluates your Big Five personality traits using advanced AI models. 
                It provides personalized insights, actionable recommendations, and guidance for personal growth, career development, 
                emotional well-being, and social relationships. The interactive dashboard and chatbot allow users to explore their personality 
                traits in depth and receive customized advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )