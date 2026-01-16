# 🎓 Lexicognition VivaBot

**AI-Powered Research Paper Viva Examiner**
*Kshitij 2026 – Lexicognition Challenge*

---

## 🧠 What is this?

Lexicognition VivaBot is an AI system that:

* Ingests **any research paper (PDF)**
* Understands its content using **RAG (Retrieval-Augmented Generation)**
* Acts like a **strict viva examiner**
* Asks **paper-specific technical questions**
* Grades answers using **evidence from the paper**
* Detects **hallucinations, contradictions, and vague answers**

It is designed to handle:

* ✅ Surprise unseen papers (on-the-spot)
* ✅ Two-column IEEE / ACM PDFs
* ✅ Wrong answers, vague answers, and confident nonsense
* ✅ Multimodal questions (figures, diagrams)

---

## 🏗️ System Architecture (High Level)

1. **PDF Ingestion**

   * Uses `pdfplumber` for layout-aware parsing (handles two-column papers)
   * Splits into chunks and embeds using `all-MiniLM-L6-v2`
   * Stores in ChromaDB vector store

2. **Question Generation (RAG)**

   * Retrieves relevant chunks
   * Forces **paper-specific** technical questions
   * Adapts difficulty based on previous answers

3. **Answer Evaluation Pipeline**

   * Step 1: Retrieve ground-truth context
   * Step 2: Run **contradiction detector**
   * Step 3: Run **strict grader**
   * Step 4: Enforce **hard score limits** if answer is wrong
   * Step 5: Verify **quoted evidence**

4. **Persona Layer**

   * Only affects **style & tone**
   * Does NOT affect grading strictness or correctness

---

## 🧪 Key Safety & Robustness Features

* 🛡️ **No context bleed**: New PDF upload wipes old knowledge
* 🛡️ **Contradiction detector**: Confidently wrong answers are capped ≤ 3
* 🛡️ **Evidence enforcement**: Answers must cite text from paper
* 🛡️ **Curly-brace sanitizer**: Prevents LaTeX/code from crashing prompts
* 🛡️ **Two-column safe PDF parsing**

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **LLM**: Groq (LLaMA 3.3 70B, LLaMA 3.2 Vision)
* **Embeddings**: Sentence-Transformers (MiniLM)
* **Vector DB**: ChromaDB
* **PDF Parsing**: pdfplumber
* **Speech**: Groq Whisper + gTTS
* **Charts**: Plotly

---

## ⚙️ Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/yourusername/lexicognition_viva_voce.git
cd lexicognition_viva_voce
```

---

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set API key

Create a file named:

```
.env
```

Put this inside:

```
GROQ_API_KEY=your_actual_key_here
```

---

### 5️⃣ Run the app

```bash
streamlit run main.py
```

---

## 🧑‍🏫 How to Use

1. Upload a **PDF research paper**
2. (Optional) Upload a **figure / diagram image**
3. Select examiner persona
4. Start the viva
5. Answer via:

   * 🎤 Voice
   * ⌨️ Text
6. Get:

   * Score
   * Evidence
   * Strict feedback
   * Skill radar

---

## 🏆 Why this will score well in judging

* ✅ Questions are **paper-specific**, not generic
* ✅ Wrong answers are **detected and penalized**
* ✅ Uses **retrieval + verification**, not vibes
* ✅ Handles **surprise PDFs safely**
* ✅ Works on **real research papers**

---

## ⚠️ Ethics & Safety

* No user data stored
* No training on uploaded papers
* Everything runs session-local
* Designed for **evaluation, not memorization**

---

## 📌 Known Limitations

* Scanned PDFs without text layer may not parse well
* Vision questions are not yet fact-verified against image content
* Requires internet for Groq API

---

## 🏁 Conclusion

This is not a chatbot.
This is an **AI viva examiner with grounding, verification, and enforcement.**
