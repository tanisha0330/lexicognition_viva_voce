import streamlit as st
import os
import tempfile
import re
import base64
import json
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image
import io

# --- 1. CONFIGURATION & CSS ---
st.set_page_config(page_title="Lexicognition VivaBot", layout="wide", page_icon="🎓")

# Custom CSS for chat bubbles
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #00CC96; }
</style>
""", unsafe_allow_html=True)

# --- 2. PERSONA SYSTEM (THE FUN PART) ---
PERSONAS = {
    "Strict Professor": {
        "icon": "👨‍🏫",
        "prompt": "You are a cynical, hard-to-please PhD Professor. You think students are lazy. Be formal, strict, and intimidating.",
        "voice": "co.uk" # British
    },
    "Roast Master": {
        "icon": "🤡",
        "prompt": "You are a savage comedian. You roast the student for every mistake. Be mean, funny, and use sarcasm. If they get it right, be begrudgingly impressed.",
        "voice": "us" # American
    },
    "Gen Z Tech Bro": {
        "icon": "🛹",
        "prompt": "You are a Gen Z tech bro. Use slang like 'no cap', 'bet', 'cooked', 'skill issue', 'L take', 'W answer'. Be chill but technical.",
        "voice": "en-ie" # Irish (closest to casual)
    },
    "Pirate Captain": {
        "icon": "🏴‍☠️",
        "prompt": "You are a Pirate Captain checking if the recruit knows the map (the paper). Use 'Arr', 'Matey', 'Landlubber'.",
        "voice": "en-au" # Australian (closest to pirate)
    }
}

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "question_count" not in st.session_state: st.session_state.question_count = 0
if "current_question" not in st.session_state: st.session_state.current_question = None
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "history" not in st.session_state: st.session_state.history = [] 
if "metrics" not in st.session_state: 
    st.session_state.metrics = {"Accuracy": [], "Depth": [], "Clarity": []}
if "vision_image_data" not in st.session_state: st.session_state.vision_image_data = None
if "last_q_type" not in st.session_state: st.session_state.last_q_type = "text"

# --- 4. HELPER FUNCTIONS ---
def text_to_speech(text, accent_tld='co.uk'):
    """Plays audio using gTTS with dynamic accents. Fixed for Windows file locking."""
    try:
        # Clean text of markdown/asterisks for better speech
        clean_text = re.sub(r'\*+', '', text) 
        tts = gTTS(text=clean_text, lang='en', tld=accent_tld)
        
        # Create a temporary file
        # delete=False is required to close the file before reopening/deleting on Windows
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_filename = fp.name
            # Close the file handle immediately so gTTS/other processes can access it freely
        
        # Write to the file
        tts.save(temp_filename)
        
        # Read the file
        with open(temp_filename, 'rb') as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
            
        # Delete the file now that we are done and it's closed
        os.remove(temp_filename)
        
    except Exception as e:
        # Log error to console but don't crash UI
        print(f"Audio Error: {e}")

def parse_json_robust(text):
    """Extracts JSON object from a string even if LLM chatters around it."""
    try:
        # Regex to find the first { and last }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return None
    except:
        return None

def plot_radar_chart():
    """Generates the Radar Chart."""
    if not st.session_state.metrics["Accuracy"]: return None
    
    # Calculate averages
    avgs = [
        sum(st.session_state.metrics[k]) / len(st.session_state.metrics[k])
        for k in ["Accuracy", "Depth", "Clarity"]
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avgs,
        theta=["Accuracy", "Depth", "Clarity"],
        fill='toself',
        line_color='#FF4B4B'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

# Load API Key
load_dotenv()

# --- 5. IMPORTS (LangChain & Groq) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from groq import Groq # Direct client for Vision

# --- 6. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.header("⚙️ Exam Settings")
    
    # Persona Selector
    selected_persona = st.selectbox("Examiner Persona", list(PERSONAS.keys()))
    persona_data = PERSONAS[selected_persona]
    
    st.divider()
    st.header("📊 Live Stats")
    
    if st.session_state.history:
        avg = sum(h['score'] for h in st.session_state.history) / len(st.session_state.history)
        st.metric("GPA", f"{avg:.1f} / 10", delta=f"{st.session_state.history[-1]['score'] - avg:.1f}")
        
        st.caption("Skill Radar")
        chart = plot_radar_chart()
        # FIXED: Replaced use_container_width=True with width="stretch" per error log
        if chart: st.plotly_chart(chart, width="stretch")
            
        st.progress(min(st.session_state.question_count / 5, 1.0))
        st.caption(f"Question {st.session_state.question_count}/5")
        
        if st.button("📄 Download Transcript"):
            report = f"TRANSCRIPT - {selected_persona} Mode\n" + "-"*30 + "\n"
            for h in st.session_state.history:
                report += f"Q: {h['question']}\nA: {h['answer']}\nScore: {h['score']}\nFeedback: {h['feedback']}\n\n"
            st.download_button("Download", report, file_name="transcript.txt")
    else:
        st.info("Waiting for exam start...")

    if st.button("🔄 Reset System"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

# --- 7. CORE LOGIC ---
st.title(f"{persona_data['icon']} Lexicognition: {selected_persona} Mode")

@st.cache_resource
def process_pdf(file_content):
    """Ingest PDF - Cached for Speed"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Clean up temp PDF file properly
    try:
        os.remove(tmp_path)
    except:
        pass
        
    return vectorstore

def generate_question(vector_store, history, persona_prompt):
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.7)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Adaptive Logic
    last_score = history[-1]['score'] if history else None
    instruction = "Ask a conceptual question about the core contribution."
    if last_score is not None:
        if last_score < 5: instruction = "Student is failing. Ask a simple definition question."
        elif last_score > 8: instruction = "Student is excelling. Ask a complex critique question."

    template = f"""{persona_prompt}
    Instruction: {instruction}
    Task: Generate ONE Viva question based on context. Keep it short.
    Context: {{context}}
    Question:"""
    
    chain = (
        {"context": retriever | (lambda d: "\n".join(x.page_content for x in d)), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )
    return chain.invoke("new question")

def generate_vision_question(image_base64, persona_prompt):
    """Generates a question from an image using Llama 3.2 Vision"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"{persona_prompt} Look at this image from the paper. Ask a hard conceptual question about this data, architecture, or diagram. Keep it short."
    
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        temperature=0.7,
        max_tokens=100
    )
    return completion.choices[0].message.content

def grade_answer(user_answer, question, vector_store, persona_prompt, image_context=None):
    api_key = os.getenv("GROQ_API_KEY")
    # If image_context is present, we technically should use vision model to grade, 
    # but Llama 3.3 text model is smarter at logic. We'll pass "Context: Image Question"
    
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.1)
    
    context_text = "N/A (Visual Question)"
    page_num = "Image"
    
    if not image_context:
        retriever = vector_store.as_retriever()
        docs = retriever.invoke(f"{question} {user_answer}")
        context_text = "\n".join([d.page_content for d in docs])
        page_num = docs[0].metadata.get('page', 'Unknown')
    
    # Updated template with quadruple braces for proper JSON handling in f-strings + LangChain
    template = f"""{persona_prompt}
    
    Question: {question}
    Student Answer: {user_answer}
    True Context (Page {page_num}): {context_text}
    
    Task: Grade the answer and return ONLY valid JSON.
    Format:
    {{{{
        "score": (int 0-10),
        "feedback": "Your persona-based response here",
        "evidence": "Short quote from text (or 'Visual Check' if image)",
        "metrics": {{{{ "Accuracy": (int), "Depth": (int), "Clarity": (int) }}}}
    }}}}
    """
    
    chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
    response_str = chain.invoke({})
    
    data = parse_json_robust(response_str)
    
    # Fallback if AI fails to give JSON
    if not data:
        data = {
            "score": 5, "feedback": "I understood that, but my grading system glitched. Let's move on.", 
            "evidence": "N/A", "metrics": {"Accuracy": 5, "Depth": 5, "Clarity": 5}
        }
    return data, page_num

# --- 8. MAIN UI ---
st.sidebar.divider()
st.sidebar.header("📂 Knowledge Base")
uploaded_file = st.sidebar.file_uploader("1. Upload PDF (Text)", type="pdf")
uploaded_image = st.sidebar.file_uploader("2. Upload Figure/Graph (Vision)", type=["png", "jpg", "jpeg"])

# Handle PDF Ingestion
if uploaded_file:
    if st.session_state.vector_store is None:
        with st.spinner("🧠 Ingesting Text..."):
            st.session_state.vector_store = process_pdf(uploaded_file.getvalue())
        st.sidebar.success("PDF Loaded!")

# Handle Image Ingestion
if uploaded_image:
    if st.session_state.vision_image_data is None:
        image = Image.open(uploaded_image)
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        st.session_state.vision_image_data = base64.b64encode(buffered.getvalue()).decode()
        st.sidebar.success("Vision Data Loaded!")

if st.session_state.vector_store or st.session_state.vision_image_data:
    # Chat Display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            avatar = persona_data['icon'] if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    # Game Loop
    # Check if we are mid-question or need a new one
    if st.session_state.current_question is None:
        if st.session_state.question_count < 5:
            # DECISION: Text Question or Vision Question?
            # If image exists, 50% chance or if specific button pressed (automated here for flow)
            use_vision = False
            if st.session_state.vision_image_data:
                # Simple toggle: If last was text, try vision. Or just random.
                # Let's alternate if both exist.
                if st.session_state.vector_store and st.session_state.last_q_type == "text":
                    use_vision = True
                elif not st.session_state.vector_store:
                    use_vision = True
            
            with st.spinner(f"{selected_persona} is thinking..."):
                if use_vision:
                    q = generate_vision_question(st.session_state.vision_image_data, persona_data['prompt'])
                    st.session_state.last_q_type = "vision"
                    q_prefix = "[Visual Analysis] "
                else:
                    q = generate_question(st.session_state.vector_store, st.session_state.history, persona_data['prompt'])
                    st.session_state.last_q_type = "text"
                    q_prefix = ""

                st.session_state.current_question = q
                st.session_state.messages.append({"role": "assistant", "content": f"**Q{st.session_state.question_count + 1}:** {q_prefix}{q}"})
                st.rerun()
        else:
            st.balloons()
            st.success("🎉 EXAM COMPLETED! Download your report from the sidebar.")

    # Input & Grading
    if st.session_state.current_question and st.session_state.question_count < 5:
        audio_val = st.audio_input("🎤 Voice Answer")
        text_val = st.chat_input("Type Answer...")
        
        final_ans = None
        if audio_val:
            with st.spinner("Transcribing..."):
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                try:
                    transcription = client.audio.transcriptions.create(
                        file=("input.wav", audio_val, "audio/wav"),
                        model="whisper-large-v3",
                        response_format="json", temperature=0.0
                    )
                    final_ans = transcription.text
                    st.info(f"🗣️ You said: {final_ans}")
                except Exception as e: st.error(f"Mic Error: {e}")
        
        if text_val: final_ans = text_val

        # C. Grading Logic
        if final_ans:
            st.session_state.messages.append({"role": "user", "content": final_ans})
            
            # Show the user message immediately (Visually)
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(final_ans)
            
            with st.spinner("Grading..."):
                is_vision_q = (st.session_state.last_q_type == "vision")
                data, page = grade_answer(
                    final_ans, 
                    st.session_state.current_question, 
                    st.session_state.vector_store, 
                    persona_data['prompt'],
                    image_context=is_vision_q
                )
                
                # Update Stats
                st.session_state.history.append({
                    "question": st.session_state.current_question, "answer": final_ans, 
                    "score": data["score"], "feedback": data["feedback"]
                })
                for k, v in data["metrics"].items(): st.session_state.metrics[k].append(v)
                
                # Output
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"""**Score: {data['score']}/10**\n\n> {data['feedback']}\n\n📖 *Evidence:* "{data['evidence']}" """
                })
                
                # Visual FX
                if data['score'] >= 8: st.balloons()
                elif data['score'] <= 4: st.toast("💀 Critical Damage!", icon="🔥")
                
                # Audio FX
                text_to_speech(data['feedback'], persona_data['voice'])
                
                st.session_state.question_count += 1
                st.session_state.current_question = None
                st.rerun()
else:
    st.info("👆 Upload a PDF or a Graph Image in the sidebar to begin.")
