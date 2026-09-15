import streamlit as st
import streamlit.components.v1 as components



def apply_ui_style():
    """Match the reference layout/typography without changing the existing colors."""
    st.markdown(
        """
        <style>
        /* Typography: rounded, clean font similar to the reference image */
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif !important;
        }

        /* Keep the existing background and text colors; only adjust typography/layout. */
        .block-container {
            max-width: 1260px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }

        /* Centered page title and subtitle */
        .hero-title {
            text-align: center;
            margin: 0 auto 0.35rem auto;
        }

        .hero-title h1 {
            font-size: 3.25rem !important;
            line-height: 1.08 !important;
            font-weight: 900 !important;
            letter-spacing: -1.5px;
            margin: 0 !important;
        }

        .hero-subtitle {
            text-align: center;
            font-size: 1.12rem !important;
            font-weight: 500 !important;
            line-height: 1.5;
            margin: 0 auto 2.4rem auto;
        }

        /* Section headings */
        h2 {
            font-size: 2rem !important;
            font-weight: 900 !important;
            letter-spacing: -0.5px;
        }

        h3 {
            font-weight: 800 !important;
        }

        /* Rounded controls like the reference */
        div[data-testid="stFileUploader"] section,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            border-radius: 14px !important;
        }

        div[data-baseweb="select"] > div {
            min-height: 54px;
        }

        /* Larger, rounded primary action */
        .stButton > button {
            min-height: 52px;
            border-radius: 16px !important;
            font-family: 'Nunito', sans-serif !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
        }

        /* Make the main action visually similar in size/spacing to the reference */
        .main-action {
            margin-top: 1rem;
            margin-bottom: 1.2rem;
        }

        /* More breathing room around Streamlit columns */
        div[data-testid="column"] {
            padding-left: 0.35rem;
            padding-right: 0.35rem;
        }

        /* Metrics and tabs use the same rounded typography */
        div[data-testid="stMetricValue"] {
            font-family: 'Nunito', sans-serif !important;
            font-weight: 900 !important;
        }

        button[data-baseweb="tab"] {
            font-family: 'Nunito', sans-serif !important;
            font-weight: 700 !important;
        }

        /* Keep alerts/cards rounded without setting their colors */
        div[data-testid="stAlert"] {
            border-radius: 14px !important;
        }

        /* Sidebar typography only; no color changes */
        section[data-testid="stSidebar"] * {
            font-family: 'Nunito', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_accent_color_js(score: int):
    """Set the dashboard accent according to the ATS score."""
    if score >= 75:
        accent_color = "#22c55e"
        glow_color = "rgba(34, 197, 94, 0.2)"
    elif score >= 55:
        accent_color = "#f59e0b"
        glow_color = "rgba(245, 158, 11, 0.2)"
    else:
        accent_color = "#ef4444"
        glow_color = "rgba(239, 68, 68, 0.2)"

    js_code = f"""
    <script>
        const root = window.parent.document.documentElement;
        root.style.setProperty('--accent-color', '{accent_color}');
        const style = window.parent.document.createElement('style');
        style.innerHTML = `
            .stProgress > div > div > div > div {{
                background-color: {accent_color} !important;
            }}
            .stButton > button[kind="primary"] {{
                background-color: {accent_color} !important;
                border-color: {accent_color} !important;
                box-shadow: 0 4px 12px {glow_color} !important;
            }}
            div[data-testid="stMetricValue"] {{
                color: {accent_color} !important;
            }}
        `;
        window.parent.document.head.appendChild(style);
    </script>
    """
    components.html(js_code, height=0, width=0)


def render_sidebar():
    st.sidebar.title("⚙️ Configuration")

    default_key = st.session_state.get("env_api_key", "")
    if not default_key:
        import os
        default_key = os.getenv("GROQ_API_KEY", "")

    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="For security, keep your API key in .env and never commit it to GitHub.",
    )

    st.sidebar.caption("Model: openai/gpt-oss-20b")
    st.sidebar.divider()
    st.sidebar.markdown("### Workflow")
    st.sidebar.markdown(
        "1. Upload resume\n"
        "2. Select role / upload JD\n"
        "3. Run ATS analysis\n"
        "4. Review decision & weaknesses\n"
        "5. Ask resume questions\n"
        "6. Generate interview\n"
        "7. Improve resume"
    )
    return groq_api_key.strip()


def _show_decision(decision):
    label = decision.get("label", "Manual Review")
    icon = decision.get("icon", "⚠️")
    message = decision.get("message", "")

    if decision.get("level") == "success":
        st.success(f"{icon} **{label}** — {message}")
    elif decision.get("level") == "error":
        st.error(f"{icon} **{label}** — {message}")
    else:
        st.warning(f"{icon} **{label}** — {message}")


def _render_skill_breakdown(scores):
    if not scores:
        st.info("No granular skill scores were returned.")
        return

    for skill, data in scores.items():
        if isinstance(data, dict):
            score = int(data.get("score", 0))
            status = data.get("status", "")
            evidence = data.get("evidence", "Not provided.")
        else:
            score = int(data)
            status = ""
            evidence = "Not provided."

        st.write(f"**{skill}** — {score}/10 {('· ' + status) if status else ''}")
        st.progress(min(score / 10, 1.0))
        with st.expander("View evidence"):
            st.write(evidence)


def render_dashboard(agent, role_requirements):
    apply_ui_style()

    st.markdown(
        """
        <div class="hero-title">
            <h1>AI-Powered Recruitment Agent</h1>
        </div>
        <div class="hero-subtitle">
            Resume parsing • JD matching • ATS scoring • RAG Q&amp;A • Interview generation • Resume improvement
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("1. Upload Candidate & Job Details")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_resume = st.file_uploader(
            "Upload Resume",
            type=["pdf", "txt"],
            help="Use a text-based PDF or TXT resume.",
        )
        selected_role = st.selectbox("Target Role", list(role_requirements.keys()))

    with col2:
        use_custom_jd = st.checkbox("Use Custom Job Description (JD)")
        uploaded_jd = None
        if use_custom_jd:
            uploaded_jd = st.file_uploader(
                "Upload Job Description",
                type=["pdf", "txt"],
                help="Upload a PDF or TXT job description.",
            )
        else:
            st.info("Default role requirements will be used for ATS matching.")

    st.markdown('<div class="main-action">', unsafe_allow_html=True)
    process_btn = st.button("🚀 Process & Analyze Resume", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if process_btn:
        if not uploaded_resume:
            st.error("Please upload a resume first.")
        elif use_custom_jd and not uploaded_jd:
            st.error("Please upload a custom job description, or uncheck the custom JD option.")
        else:
            try:
                with st.spinner("Parsing documents and building the resume knowledge base..."):
                    agent.extract_text_from_file(uploaded_resume)
                    if use_custom_jd:
                        agent.extract_jd_text(uploaded_jd)
                    jd_data = agent.build_jd(selected_role, custom_jd=use_custom_jd)

                with st.spinner("Running ATS matching and weakness analysis..."):
                    result = agent.analyze_skills(jd_data, role=selected_role)
                    decision = agent.recruitment_decision(result)

                st.session_state.resume_processed = True
                st.session_state.selected_role = selected_role
                st.session_state.using_custom_jd = use_custom_jd
                st.session_state.jd_data = jd_data
                st.session_state.analysis_result = result
                st.session_state.decision = decision
                st.session_state.generated_questions = []
                st.session_state.generated_suggestions = []
                st.success("Analysis completed successfully.")
            except Exception as exc:
                st.error("Analysis failed. Check the uploaded files and Groq API configuration.")
                st.exception(exc)

    if not st.session_state.get("resume_processed"):
        st.divider()
        st.info("Upload a resume and click **Process & Analyze Resume** to start the workflow.")
        return

    selected_role = st.session_state.get("selected_role", selected_role)
    results = st.session_state.get("analysis_result") or {}
    decision = st.session_state.get("decision") or agent.recruitment_decision(results)
    overall_score = int(results.get("overall_score", 0))
    apply_accent_color_js(overall_score)

    st.divider()
    st.header("2. Candidate Evaluation")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("ATS Match Score", f"{overall_score}%")
    metric2.metric("Skills Evaluated", len(results.get("scores", {})))
    metric3.metric("Missing Keywords", len(results.get("missing_keywords", [])))
    _show_decision(decision)

    st.caption(decision.get("reason", ""))

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 ATS Analysis",
        "💬 Candidate Q&A",
        "❓ Interview Generator",
        "💡 Resume Improvements",
    ])

    with tab1:
        st.subheader("Skill Breakdown")
        _render_skill_breakdown(results.get("scores", {}))

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("✅ Candidate Strengths")
            strengths = results.get("strengths", [])
            if strengths:
                for strength in strengths:
                    st.success(strength)
            else:
                st.info("No strengths were returned.")

        with col_b:
            st.subheader("⚠️ Gaps & Missing Keywords")
            gaps = results.get("gaps", [])
            missing = results.get("missing_keywords", [])
            for gap in gaps:
                st.warning(gap)
            if missing:
                st.write("**Missing keywords:**")
                st.write(" • ".join(missing))
            if not gaps and not missing:
                st.info("No major gaps were returned.")

        st.divider()
        st.subheader("🔎 Detailed Weakness Analysis")
        weaknesses = results.get("weaknesses", [])
        if weaknesses:
            for weakness in weaknesses:
                skill = weakness.get("skill", "Skill")
                score = weakness.get("score", 0)
                severity = weakness.get("severity", "Medium")
                with st.expander(f"{skill} — {score}/10 · {severity}"):
                    st.write(f"**Evidence:** {weakness.get('evidence', 'Not provided.')}")
                    st.write(f"**Recommendation:** {weakness.get('recommendation', 'Practice and build evidence in this area.')}")
        else:
            st.success("No detailed weaknesses were identified.")

        st.divider()
        with st.expander("View Parsed Job Requirements"):
            st.json(st.session_state.get("jd_data", {}))

    with tab2:
        st.subheader("Ask Questions About the Candidate")
        st.caption("Answers are grounded in the uploaded resume. The agent will say when information is not mentioned.")
        user_query = st.text_input(
            "Question",
            placeholder="e.g., What projects demonstrate the candidate's Python experience?",
            key="candidate_question",
        )

        if st.button("🔍 Ask", key="ask_resume"):
            if not user_query.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    with st.spinner("Searching resume context..."):
                        qa = agent.answer_question(user_query.strip())
                    st.markdown("### Answer")
                    st.write(qa.get("answer", "Not mentioned in the uploaded resume."))
                    st.caption(f"Confidence: {qa.get('confidence', 'Low')}")
                    evidence = qa.get("evidence", [])
                    if evidence:
                        st.markdown("**Resume evidence:**")
                        for item in evidence:
                            st.info(item)
                except Exception as exc:
                    st.error("Could not answer the question.")
                    st.exception(exc)

    with tab3:
        st.subheader("Personalized Interview Question Generator")
        c1, c2, c3 = st.columns(3)
        with c1:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        with c2:
            question_type = st.selectbox("Question Type", ["Mixed", "Technical", "Behavioral", "Scenario", "Project", "HR"])
        with c3:
            count = st.slider("Number of Questions", 3, 10, 5)

        include_answers = st.checkbox("Include ideal answers", value=True)

        if st.button("🎤 Generate Interview Questions", type="primary", key="generate_interview"):
            try:
                with st.spinner("Generating resume-specific interview questions..."):
                    questions = agent.generate_interview_questions(
                        role=selected_role,
                        difficulty=difficulty,
                        question_type=question_type,
                        count=count,
                        include_answers=include_answers,
                    )
                st.session_state.generated_questions = questions
            except Exception as exc:
                st.error("Could not generate interview questions.")
                st.exception(exc)

        questions = st.session_state.get("generated_questions", [])
        if questions:
            for i, question in enumerate(questions, 1):
                qtext = question.get("question", "Interview question")
                qtype = question.get("type", question_type)
                qdifficulty = question.get("difficulty", difficulty)
                with st.expander(f"Question {i}: {qtext}"):
                    st.caption(f"{qtype} · {qdifficulty}")
                    if include_answers and question.get("answer"):
                        st.markdown("**Ideal Model Answer**")
                        st.write(question.get("answer"))
                    if question.get("reason"):
                        st.markdown("**Why this question?**")
                        st.write(question.get("reason"))
        else:
            st.info("Generate questions to see the personalized interview set.")

    with tab4:
        st.subheader("Actionable Resume Improvement Plan")
        st.caption("Suggestions are based on the target role, resume evidence and ATS gaps.")

        if st.button("💡 Generate Improvement Plan", type="primary", key="generate_improvements"):
            try:
                with st.spinner("Analyzing resume sections and ATS gaps..."):
                    suggestions = agent.suggest_improvements(role=selected_role)
                st.session_state.generated_suggestions = suggestions
            except Exception as exc:
                st.error("Could not generate improvement suggestions.")
                st.exception(exc)

        suggestions = st.session_state.get("generated_suggestions", [])
        if suggestions:
            for suggestion in suggestions:
                priority = suggestion.get("priority", "Medium")
                section = suggestion.get("section", "Other")
                with st.expander(f"{section} · {priority} priority"):
                    st.write(f"**Issue:** {suggestion.get('issue', 'Not specified.')}")
                    st.write(f"**Recommendation:** {suggestion.get('recommendation', 'Not specified.')}")
                    if suggestion.get("example"):
                        st.markdown("**Example:**")
                        st.info(suggestion.get("example"))
        else:
            st.info("Generate an improvement plan to see section-wise recommendations.")
