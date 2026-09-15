#  AI Recruitment Agent

An AI-powered recruitment assistant that analyzes resumes against job requirements and helps recruiters make faster, explainable hiring decisions.

## 🚀 Features

* 📄 **Resume Parsing** – Extracts text from PDF/TXT resumes.
* 💼 **Job Description Analysis** – Supports predefined roles and custom JDs.
* 📊 **ATS Matching** – Generates skill-wise scores and an overall match score.
* 🎯 **Recruitment Decision** – Recommended, Manual Review, or Not Recommended.
* 🔍 **Weakness Analysis** – Identifies skill gaps and provides recommendations.
* 💬 **Resume Q&A** – Ask questions using resume-grounded RAG.
* 🎤 **Interview Generator** – Generates personalized interview questions.
* 💡 **Resume Improvement** – Provides actionable ATS improvement suggestions.

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Groq LLM
* FAISS
* Hugging Face Embeddings
* PyPDF2

## 📁 Project Structure

```text
AI-Recruitment-Agent/
├── app.py
├── agents.py
├── ui.py
├── requirements.txt
├── Dockerfile
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

## ▶️ Run

```bash
streamlit run app.py
```

## 🔄 Workflow

```text
Upload Resume
      ↓
Select Role / Upload JD
      ↓
ATS Analysis
      ↓
Candidate Evaluation
      ↓
Resume Q&A
      ↓
Interview Generation
      ↓
Resume Improvement
```

## ⚠️ Note

ATS scores are AI-generated recommendations and should support, not replace, human recruitment decisions.
