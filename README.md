# Lightweight Diagnostic Failure Detection (RAG System)

This repository contains a local prototype of a Retrieval-Augmented Generation (RAG) system with a **Lightweight Failure Detection Framework**.

It evaluates three separate signals (`sr`, `so`, `sn`) to detect hallucinations and retrieval failures, dynamically utilizing an Adaptive Control Policy to decide whether to:
1. **Finalize** the generated answer.
2. **Re-retrieve** documents and generate again.
3. **Abstain** to prevent hallucination.

---

## 🚀 How to Run on a New Machine (or your Friend's Laptop)

If you downloaded this project as a ZIP or cloned it from GitHub, you might notice missing folders like `__pycache__`, `faiss_index`, or `.env`. **This is normal!** These files are specific to your machine and are ignored by Git.

To run the application successfully on a new computer, follow these **5 exact steps**:

### Step 1: Install Python and Dependencies
Ensure Python 3.9+ is installed. Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Set up Environment Variables
The `.env` file contains your API keys, and because it's a security risk, it is never included in ZIPs or GitHub. 
1. Duplicate the `.env.example` file.
2. Rename the duplicate to exactly `.env` (with the dot in front).
3. Open `.env` and paste in your Gemini API Key.
*(Leave `USE_MOCK_LLM=True` if you don't have an API key and just want to see the demo).*

### Step 3: Build the Local Vector Database (FAISS Index)
The AI needs to read your documents before it can answer questions.
```bash
python ingest.py
```
*(This will read the text files in `data/documents/` and generate the `faiss_index` folder).*

### Step 4: Start the API Server
In your terminal, start the FastAPI backend:
```bash
python main.py
```
Wait until you see `Application startup complete` in the console. Do not close this terminal!

### Step 5: Start the User Interface
Open **a second, new terminal window** in the same folder and run:
```bash
streamlit run streamlit_app.py
```
This will automatically open the UI in your web browser.

---

## 🎯 How to Demonstrate the Project to Reviewers

We have built a special script that proves the system can execute all 3 logic branches perfectly.

**While the API server (`main.py`) is running**, open a terminal and run:

```bash
python demo_scenarios.py
```

It will execute three tests automatically and print the internal metrics:
1. **Scenario 1 (Finalize):** "What is quantum computing?" → High similarity, High overlap, High exactness. System answers.
2. **Scenario 2 (Abstain):** "How do I bake a chocolate cake?" → Low similarity (out of domain). System refuses to answer to prevent hallucination.
3. **Scenario 3 (Generation Failure):** "hallucinate quantum computing" → High similarity, but the text is forced to hallucinate. System catches the low NLI and overlap score, and decides to Re-retrieve.

---

## 📁 Technical Architecture
- **Web UI:** Streamlit
- **Backend App:** FastAPI
- **Vector Search:** FAISS (`IndexFlatIP`)
- **Embeddings Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **NLI Model:** `nli-deberta-v3-small` (Cross-encoder)
- **LLM Generator:** Google Gemini 2.0 Flash (`google-genai`)
