import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseAgent:
    """Shared LLM helper used by the recruitment agents."""

    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def call_llm(self, prompt: str, max_retries: int = 4) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                content = getattr(response, "content", response)
                return str(content).strip()
            except Exception as exc:
                last_error = exc
                if "429" in str(exc) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f"LLM request failed: {last_error}")

    @staticmethod
    def parse_json(text: str, fallback: Any) -> Any:
        """Parse JSON even when the model wraps it in markdown/code fences."""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find the first JSON object/array in the response.
        starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
        if not starts:
            return fallback

        start = min(starts)
        opening = cleaned[start]
        closing = "}" if opening == "{" else "]"
        end = cleaned.rfind(closing)
        if end <= start:
            return fallback

        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return fallback


class ResumeParserAgent:
    """Extracts and normalizes candidate resume text."""

    def extract_text(self, file_obj) -> str:
        if not file_obj:
            raise ValueError("No resume file was provided.")

        name = getattr(file_obj, "name", "resume.txt").lower()
        file_obj.seek(0)

        if name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file_obj)
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(pages)
        else:
            raw = file_obj.read()
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="ignore")
            else:
                text = str(raw)

        text = self.clean_text(text)
        if not text:
            raise ValueError(
                "No readable text was found in the resume. "
                "Try a text-based PDF or TXT file."
            )
        return text

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class JobDescriptionAgent(BaseAgent):
    """Extracts text and structured requirements from a job description."""

    def __init__(self, llm: ChatGroq, role_requirements: Dict[str, List[str]]):
        super().__init__(llm)
        self.role_requirements = role_requirements

    def extract_jd_text(self, file_obj) -> str:
        return ResumeParserAgent().extract_text(file_obj)

    def parse_jd(self, jd_text: str, role: str) -> Dict[str, Any]:
        prompt = f"""
You are a job-description analysis agent.

Target role: {role}

Job description:
{jd_text}

Extract only information supported by the job description. Return ONLY valid JSON:
{{
  "required_skills": ["skill 1", "skill 2"],
  "preferred_skills": ["skill 1"],
  "experience_requirements": ["requirement"],
  "education_requirements": ["requirement"],
  "keywords": ["keyword 1"],
  "responsibilities": ["responsibility 1"]
}}
Do not invent requirements.
"""
        fallback = {
            "required_skills": [],
            "preferred_skills": [],
            "experience_requirements": [],
            "education_requirements": [],
            "keywords": [],
            "responsibilities": [],
        }
        result = self.parse_json(self.call_llm(prompt), fallback)
        return result if isinstance(result, dict) else fallback

    def default_jd(self, role: str) -> Dict[str, Any]:
        skills = self.role_requirements.get(role, [])
        return {
            "required_skills": skills,
            "preferred_skills": [],
            "experience_requirements": [],
            "education_requirements": [],
            "keywords": skills,
            "responsibilities": [],
        }


class ATSMatchingAgent(BaseAgent):
    """Performs explainable ATS matching using the resume and structured JD."""

    def analyze(self, resume_text: str, jd: Dict[str, Any], role: str) -> Dict[str, Any]:
        required = jd.get("required_skills", [])
        preferred = jd.get("preferred_skills", [])
        keywords = jd.get("keywords", [])

        prompt = f"""
You are an explainable ATS matching agent.

Target role: {role}

Candidate resume:
{resume_text}

Structured job requirements:
{json.dumps(jd, ensure_ascii=False, indent=2)}

Evaluate the candidate only from information explicitly present in the resume.
Do not assume a skill is present because it is related to another skill.
For every required skill, give a 0-10 score and evidence.

Scoring guidance:
0 = absent
1-3 = weak/indirect evidence
4-6 = some evidence/basic proficiency
7-8 = good evidence
9-10 = strong/direct evidence

Return ONLY valid JSON in this exact shape:
{{
  "scores": {{
    "Python": {{
      "score": 8,
      "status": "Strong",
      "evidence": "Short resume evidence"
    }}
  }},
  "overall_score": 0,
  "strengths": ["..."],
  "gaps": ["..."],
  "weaknesses": [
    {{
      "skill": "Skill name",
      "score": 2,
      "severity": "High",
      "evidence": "Why the resume is weak or missing",
      "recommendation": "Actionable recommendation"
    }}
  ],
  "keyword_matches": ["..."],
  "missing_keywords": ["..."]
}}

Calculate overall_score as a realistic 0-100 match score. Required skills should
matter more than preferred skills. Do not give a high score merely because the
resume contains generic buzzwords.
"""
        fallback = {
            "scores": {},
            "overall_score": 0,
            "strengths": [],
            "gaps": [],
            "weaknesses": [],
            "keyword_matches": [],
            "missing_keywords": [],
        }
        result = self.parse_json(self.call_llm(prompt), fallback)
        if not isinstance(result, dict):
            result = fallback

        # Normalize the model output so the UI can safely consume it.
        normalized_scores: Dict[str, Dict[str, Any]] = {}
        for skill, value in (result.get("scores") or {}).items():
            if isinstance(value, dict):
                score = self._clamp_score(value.get("score", 0))
                normalized_scores[str(skill)] = {
                    "score": score,
                    "status": value.get("status", self.score_status(score)),
                    "evidence": str(value.get("evidence", "Not provided.")),
                }
            else:
                score = self._clamp_score(value)
                normalized_scores[str(skill)] = {
                    "score": score,
                    "status": self.score_status(score),
                    "evidence": "Evidence not provided by the model.",
                }

        result["scores"] = normalized_scores
        result["overall_score"] = max(0, min(100, int(result.get("overall_score", 0))))
        result["strengths"] = [str(x) for x in (result.get("strengths") or [])]
        result["gaps"] = [str(x) for x in (result.get("gaps") or [])]
        result["missing_keywords"] = [str(x) for x in (result.get("missing_keywords") or [])]
        result["keyword_matches"] = [str(x) for x in (result.get("keyword_matches") or [])]
        result["weaknesses"] = self._normalize_weaknesses(result.get("weaknesses"))

        # If the model did not provide a weakness entry, derive it from low skill scores.
        if not result["weaknesses"]:
            for skill, data in normalized_scores.items():
                score = data["score"]
                if score <= 6:
                    result["weaknesses"].append(
                        {
                            "skill": skill,
                            "score": score,
                            "severity": "High" if score <= 3 else "Medium",
                            "evidence": data["evidence"],
                            "recommendation": f"Build stronger practical evidence for {skill} and add it to the resume if applicable.",
                        }
                    )
        return result

    @staticmethod
    def _clamp_score(value: Any) -> int:
        try:
            return max(0, min(10, int(float(value))))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def score_status(score: int) -> str:
        if score >= 8:
            return "Strong"
        if score >= 5:
            return "Moderate"
        return "Weak"

    def _normalize_weaknesses(self, weaknesses: Any) -> List[Dict[str, Any]]:
        output = []
        for item in weaknesses or []:
            if not isinstance(item, dict):
                continue
            score = self._clamp_score(item.get("score", 0))
            output.append(
                {
                    "skill": str(item.get("skill", "Unknown skill")),
                    "score": score,
                    "severity": str(item.get("severity", "Medium")),
                    "evidence": str(item.get("evidence", "Not provided.")),
                    "recommendation": str(item.get("recommendation", "Practice this area and add relevant evidence when available.")),
                }
            )
        return output

    @staticmethod
    def decision(score: int) -> Dict[str, str]:
        if score >= 75:
            return {"label": "Recommended", "icon": "✅", "message": "Candidate is strongly aligned with the target role.", "level": "success"}
        if score >= 55:
            return {"label": "Manual Review", "icon": "⚠️", "message": "Candidate has relevant evidence but should be reviewed by a recruiter.", "level": "warning"}
        return {"label": "Not Recommended", "icon": "❌", "message": "The resume does not currently provide enough evidence for the target requirements.", "level": "error"}


class CandidateQAAgent(BaseAgent):
    """Resume-grounded Q&A using FAISS retrieval."""

    def __init__(self, llm: ChatGroq, embeddings: HuggingFaceEmbeddings):
        super().__init__(llm)
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self.resume_text = ""

    def index_resume(self, text: str) -> None:
        self.resume_text = text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        if not chunks:
            self.vector_store = None
            return
        self.vector_store = FAISS.from_documents(
            [Document(page_content=chunk) for chunk in chunks],
            self.embeddings,
        )

    def answer(self, question: str) -> Dict[str, Any]:
        if not self.vector_store:
            return {"answer": "Please upload and process a resume first.", "confidence": "Low", "evidence": []}

        docs = self.vector_store.similarity_search(question, k=4)
        context = "\n\n--- RESUME SECTION ---\n\n".join(doc.page_content for doc in docs)
        prompt = f"""
You are a recruitment resume-Q&A agent.
Answer the user's question using ONLY the retrieved resume context below.
If the answer is not supported by the context, say exactly:
"Not mentioned in the uploaded resume."
Do not invent dates, employers, skills, experience, education, or achievements.

Retrieved resume context:
{context}

Question: {question}

Return ONLY valid JSON:
{{
  "answer": "...",
  "confidence": "High|Medium|Low",
  "evidence": ["short supporting phrase 1", "short supporting phrase 2"]
}}
"""
        fallback = {"answer": "Not mentioned in the uploaded resume.", "confidence": "Low", "evidence": []}
        result = self.parse_json(self.call_llm(prompt), fallback)
        return result if isinstance(result, dict) else fallback


class InterviewAgent(BaseAgent):
    """Generates personalized interview questions from resume + ATS gaps."""

    def generate(
        self,
        role: str,
        resume_text: str,
        ats_result: Optional[Dict[str, Any]] = None,
        difficulty: str = "Medium",
        question_type: str = "Mixed",
        count: int = 5,
        include_answers: bool = True,
    ) -> List[Dict[str, str]]:
        ats_result = ats_result or {}
        prompt = f"""
You are an AI interview-generation agent.

Role: {role}
Difficulty: {difficulty}
Question type: {question_type}
Number of questions: {count}
Include ideal answers: {include_answers}

Candidate resume:
{resume_text}

ATS analysis:
{json.dumps(ats_result, ensure_ascii=False, indent=2)}

Create personalized interview questions. Use the candidate's actual projects,
skills, strengths and gaps when possible. Do not invent candidate experience.
Return ONLY a JSON array:
[
  {{
    "question": "...",
    "answer": "...",
    "type": "Technical|Behavioral|Scenario|Project|HR",
    "difficulty": "Easy|Medium|Hard",
    "reason": "Why this question is relevant to this candidate"
  }}
]
"""
        fallback: List[Dict[str, str]] = []
        result = self.parse_json(self.call_llm(prompt), fallback)
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)][:count]


class ResumeImprovementAgent(BaseAgent):
    """Creates section-wise, actionable resume improvement suggestions."""

    def suggest(self, resume_text: str, role: str, ats_result: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        prompt = f"""
You are a resume improvement agent.

Target role: {role}
Candidate resume:
{resume_text}

ATS analysis:
{json.dumps(ats_result or {}, ensure_ascii=False, indent=2)}

Give practical suggestions that improve ATS compatibility and recruiter clarity.
Never tell the candidate to add a skill they do not have. If a skill is missing,
suggest learning it or adding it only if they genuinely have the experience.

Return ONLY valid JSON:
[
  {{
    "section": "Summary|Skills|Projects|Experience|Education|Certifications|Formatting|Other",
    "priority": "High|Medium|Low",
    "issue": "Specific issue",
    "recommendation": "Specific action",
    "example": "Optional example wording or structure"
  }}
]
"""
        result = self.parse_json(self.call_llm(prompt), [])
        return [item for item in result if isinstance(item, dict)] if isinstance(result, list) else []


class RecruitmentDecisionAgent(BaseAgent):
    """Converts ATS evidence into a transparent recruitment decision."""

    def decide(self, ats_result: Dict[str, Any]) -> Dict[str, Any]:
        score = max(0, min(100, int(ats_result.get("overall_score", 0))))
        decision = ATSMatchingAgent.decision(score)
        return {
            "score": score,
            **decision,
            "reason": self._reason(ats_result),
        }

    @staticmethod
    def _reason(result: Dict[str, Any]) -> str:
        scores = result.get("scores") or {}
        if not scores:
            return "Decision is based on the overall ATS match score."
        strong = sum(1 for v in scores.values() if isinstance(v, dict) and v.get("score", 0) >= 8)
        weak = sum(1 for v in scores.values() if isinstance(v, dict) and v.get("score", 0) <= 4)
        return f"The evaluation found {strong} strong skill area(s) and {weak} weak skill area(s) among the evaluated requirements."


class ResumeAnalysisAgent:
    """Backward-compatible facade coordinating all recruitment agents."""

    def __init__(self, api_key: str, role_requirements: Optional[Dict[str, List[str]]] = None):
        if not api_key or not api_key.strip():
            raise ValueError("A Groq API key is required.")

        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.llm = ChatGroq(
            groq_api_key=api_key.strip(),
            model=model,
            temperature=0.2,
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.role_requirements = role_requirements or {}
        self.resume_parser = ResumeParserAgent()
        self.jd_agent = JobDescriptionAgent(self.llm, self.role_requirements)
        self.ats_agent = ATSMatchingAgent(self.llm)
        self.qa_agent = CandidateQAAgent(self.llm, self.embeddings)
        self.interview_agent = InterviewAgent(self.llm)
        self.improvement_agent = ResumeImprovementAgent(self.llm)
        self.decision_agent = RecruitmentDecisionAgent(self.llm)

        self.resume_text = ""
        self.jd_text = ""
        self.jd_data: Dict[str, Any] = {}
        self.last_ats_result: Dict[str, Any] = {}

    def extract_text_from_file(self, file_obj) -> str:
        self.resume_text = self.resume_parser.extract_text(file_obj)
        self.qa_agent.index_resume(self.resume_text)
        return self.resume_text

    def extract_jd_text(self, file_obj) -> str:
        self.jd_text = self.jd_agent.extract_jd_text(file_obj)
        return self.jd_text

    def build_jd(self, role: str, custom_jd: bool = False) -> Dict[str, Any]:
        if custom_jd and self.jd_text:
            self.jd_data = self.jd_agent.parse_jd(self.jd_text, role)
        else:
            self.jd_data = self.jd_agent.default_jd(role)
        return self.jd_data

    def analyze_skills(self, required_skills: List[str] | Dict[str, Any], role: str = "") -> Dict[str, Any]:
        if isinstance(required_skills, dict):
            jd = required_skills
        else:
            jd = {
                "required_skills": required_skills,
                "preferred_skills": [],
                "experience_requirements": [],
                "education_requirements": [],
                "keywords": required_skills,
                "responsibilities": [],
            }
        self.last_ats_result = self.ats_agent.analyze(self.resume_text, jd, role)
        return self.last_ats_result

    def recruitment_decision(self, ats_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.decision_agent.decide(ats_result or self.last_ats_result)

    def answer_question(self, question: str) -> Dict[str, Any]:
        return self.qa_agent.answer(question)

    def generate_interview_questions(
        self,
        role: str,
        difficulty: str = "Medium",
        question_type: str = "Mixed",
        count: int = 5,
        include_answers: bool = True,
    ) -> List[Dict[str, str]]:
        return self.interview_agent.generate(
            role=role,
            resume_text=self.resume_text,
            ats_result=self.last_ats_result,
            difficulty=difficulty,
            question_type=question_type,
            count=count,
            include_answers=include_answers,
        )

    def suggest_improvements(self, role: str = "") -> List[Dict[str, str]]:
        return self.improvement_agent.suggest(
            resume_text=self.resume_text,
            role=role,
            ats_result=self.last_ats_result,
        )
