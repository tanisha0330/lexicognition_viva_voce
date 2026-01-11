🎓 Lexicognition VivaBot

"Redefining Education" — The Flagship AI Entry for Kshitij 2026

Problem Statement: Build an AI Viva Voce Examiner.

# Overview

Lexicognition VivaBot is not just a PDF summarizer; it is a relentless, intelligent External Examiner.

Designed for the "Lexicognition" challenge, this AI agent ingests research papers (and visual diagrams) to conduct a high-stakes, simulated oral examination. It challenges students with conceptual questions, adapts to their skill level, and grades them strictly against the source material using RAG (Retrieval-Augmented Generation).

# Key Features (The "Wow" Factors)

- Dynamic Persona System: * Strict Professor: Formal, intimidating, and academic.

Roast Master: Uses sarcasm and wit to critique wrong answers.

Gen Z Tech Bro: Uses slang ("no cap", "cooked") for a casual vibe.

Pirate Captain: A fun, thematic mode for engagement.

- Adaptive Difficulty Engine: The AI remembers your session history. If you fail, it asks fundamental definition questions. If you excel, it shifts to complex "Critique" and "Analysis" questions.

- Multimodal Vision Analysis: Supports uploading figures/graphs. The bot uses Llama 3.2 Vision to ask specific questions about data visualization.

- Full Voice Interaction:

Input: Ultra-fast voice transcription using Groq Whisper.

Output: Dynamic Text-to-Speech (gTTS) with accents matching the persona (e.g., British for the Professor).

- Real-Time Skill Analytics: A sidebar "Skill Radar" chart tracks your Accuracy, Critical Thinking, and Clarity live during the exam.

- Fact-Checked Grading: Every evaluation includes a strict Verdict, a Score (0-10), and a Citation (Page Number/Quote) from the paper to prove the AI isn't hallucinating.

🛠️ Tech Stack

Groq API (Llama-3.3-70b)

Llama-3.2-11b-vision

HuggingFace

all-MiniLM-L6-v2 (Local & Free).

Vector DB

ChromaDB

Streamlit

Audio

Groq Whisper & gTTS

Speech-to-Text & Text-to-Speech.

Plotly

# Installation & Setup

Prerequisites

Python 3.10+

A Groq API Key (Free)

1. Clone the Repository

git clone [https://github.com/tanisha0330/lexicognition-vivabot.git](https://github.com/yourusername/lexicognition-vivabot.git)
cd lexicognition-vivabot


2. Create a Virtual Environment

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install streamlit langchain langchain-groq langchain-community chromadb pypdf python-dotenv sentence-transformers plotly gTTS pillow


4. Configure API Keys

Create a file named .env in the root directory and add your key:

GROQ_API_KEY=gsk_your_actual_api_key_here


5. Run the Application

streamlit run app.py


# How to Use

Upload Knowledge: Use the sidebar to upload a Research Paper (PDF). Optionally, upload a Graph/Chart image for vision testing.

Select Examiner: Choose a persona (e.g., "Strict Professor") from the sidebar settings.

Start the Viva: The AI will generate a conceptual question.

Answer: * Voice: Click "Record Answer" and speak.

Text: Type your answer in the chat box.

Review Feedback: Watch the AI grade you live, update your Radar Chart, and verbally respond.

Download Report: At the end of the session, download your full Viva Transcript.

#Architecture Flowchart

graph TD
    User[User] -->|Uploads PDF| Ingest{Ingestion Pipeline}
    Ingest -->|PyPDFLoader| Text[Raw Text]
    Text -->|RecursiveSplitter| Chunks[Text Chunks]
    Chunks -->|HuggingFace Embeddings| VectorDB[(ChromaDB)]
    
    User -->|Selects Persona| System[System Prompt]
    
    subgraph "Exam Loop"
        GenQ{Generate Question} -->|Retrieve Context| VectorDB
        VectorDB -->|Top-k Context| GenQ
        GenQ -->|Llama 3.3| Q[Question]
        
        Q --> User
        User -->|Voice/Text Answer| Grading{Grading Agent}
        
        Grading -->|Retrieve Evidence| VectorDB
        Grading -->|Compare Answer vs PDF| Eval[Llama 3.3 Evaluation]
        
        Eval -->|JSON Output| Dashboard[Update Stats/Radar]
        Eval -->|TTS Audio| User
    end
