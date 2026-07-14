import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from database import initialize_database, save_prediction

model = joblib.load("student_model.pkl")
features = joblib.load("features.pkl")

initialize_database()

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "student_data" not in st.session_state:
    st.session_state.student_data = {}

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    html, body, [data-testid="stAppViewContainer"] {
        margin: 0 !important;
        padding: 0 !important;
        background: #07132f;
    }

    [data-testid="stAppViewContainer"] {
        background:
        radial-gradient(
            circle at 85% 75%,
            rgba(147, 51, 234, 0.28),
            transparent 35%
        ),
        radial-gradient(
            circle at 20% 10%,
            rgba(37, 99, 235, 0.20),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #07132f 0%,
            #101b51 50%,
            #25125d 100%
        );
        min-height: 100vh;
    }

    [data-testid="stHeader"] {
        background: transparent;
        height: 0px;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    [data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #07142f 0%,
            #0b1839 100%
        );
        border-right: 1px solid rgba(130, 150, 255, 0.20);
    }

    [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1rem !important;
    }

    .sidebar-brand {
        text-align: center;
        padding: 15px 5px 20px 5px;
    }

    .sidebar-icon {
        font-size: 55px;
    }

    .sidebar-title {
        font-size: 21px;
        font-weight: 750;
        color: white;
        margin-top: 10px;
    }

    .sidebar-gradient {
        font-size: 22px;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #d946ef,
            #8b5cf6
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar-subtitle {
        color: #a7b5dc;
        font-size: 13px;
        margin-top: 8px;
    }

    [data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        height: 48px;
        text-align: left;
        justify-content: flex-start;
        border: none;
        border-radius: 10px;
        padding-left: 18px;
        margin-bottom: 4px;
        background: transparent;
        color: #d7def4;
        font-size: 15px;
        font-weight: 500;
    }

    [data-testid="stSidebar"] div.stButton > button:hover {
        background: linear-gradient(
            90deg,
            rgba(124, 58, 237, 0.75),
            rgba(79, 70, 229, 0.65)
        );
        color: white;
        border: none;
    }

    .page-title {
        font-size: 45px;
        font-weight: 800;
        color: white;
        line-height: 1.2;
        margin-top: 5px;
        margin-bottom: 8px;
    }

    .gradient-title {
        background: linear-gradient(
            90deg,
            #c04dff,
            #4e8cff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .page-subtitle {
        color: #bdc9eb;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .glass-card {
        background: rgba(7, 21, 65, 0.72);
        border: 1px solid rgba(120, 145, 255, 0.24);
        border-radius: 17px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.20);
    }

    .card-title {
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .info-card {
        background: rgba(13, 30, 79, 0.72);
        border: 1px solid rgba(116, 141, 255, 0.22);
        border-radius: 15px;
        padding: 20px;
        height: 100%;
    }

    .info-value {
        font-size: 34px;
        font-weight: 800;
        color: white;
    }

    .info-label {
        color: #aebce2;
        font-size: 14px;
    }

    .score-circle {
        width: 205px;
        height: 205px;
        border-radius: 50%;
        margin: auto;
        background:
        radial-gradient(
            circle,
            #101b52 59%,
            transparent 60%
        ),
        conic-gradient(
            #4ade80 0deg 300deg,
            rgba(255,255,255,0.13) 300deg
        );
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .score-number {
        font-size: 42px;
        color: white;
        font-weight: 800;
    }

    .score-small {
        color: #b8c4e3;
        font-size: 16px;
    }

    .success-box {
        background: rgba(34, 197, 94, 0.12);
        border-left: 4px solid #4ade80;
        border-radius: 8px;
        padding: 12px;
        color: #dcfce7;
        margin-bottom: 9px;
    }

    .warning-box {
        background: rgba(251, 191, 36, 0.12);
        border-left: 4px solid #facc15;
        border-radius: 8px;
        padding: 12px;
        color: #fef3c7;
        margin-bottom: 9px;
    }

    .danger-box {
        background: rgba(244, 63, 94, 0.12);
        border-left: 4px solid #fb7185;
        border-radius: 8px;
        padding: 12px;
        color: #ffe4e6;
        margin-bottom: 9px;
    }

    .recommend-box {
        background: rgba(56, 189, 248, 0.11);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 13px;
        color: #e0f2fe;
        margin-bottom: 10px;
    }

    .welcome-card {
        background:
        linear-gradient(
            120deg,
            rgba(124, 58, 237, 0.30),
            rgba(37, 99, 235, 0.24)
        );
        border: 1px solid rgba(150, 160, 255, 0.28);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    }

    .welcome-title {
        color: white;
        font-size: 28px;
        font-weight: 750;
    }

    .welcome-text {
        color: #c9d3ee;
        font-size: 16px;
        margin-top: 8px;
    }

    .quote-card {
        background: rgba(17, 31, 78, 0.75);
        border: 1px solid rgba(120, 150, 255, 0.20);
        border-radius: 15px;
        padding: 18px;
        text-align: center;
        margin-top: 25px;
    }

    .quote-text {
        color: #d8b4fe;
        font-weight: 700;
        font-size: 16px;
    }

    .quote-sub {
        color: #9faed5;
        font-size: 12px;
    }

    div.stButton > button {
        width: 100%;
        border: none;
        color: white;
        font-weight: 700;
        border-radius: 11px;
        background: linear-gradient(
            90deg,
            #a338ed,
            #5145e5,
            #22b7d9
        );
    }

    div.stButton > button:hover {
        color: white;
        border: none;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.45);
    }

    .stTextInput input,
    .stNumberInput input {
        background: rgba(21, 38, 88, 0.78);
        color: white;
        border: 1px solid rgba(140, 160, 255, 0.30);
        border-radius: 9px;
    }

    label,
    [data-testid="stWidgetLabel"] {
        color: #d1daf2 !important;
    }

    [data-testid="stMetric"] {
        background: rgba(12, 27, 73, 0.76);
        border: 1px solid rgba(120, 145, 255, 0.20);
        padding: 16px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #b9c4e4;
    }

    [data-testid="stMetricValue"] {
        color: white;
    }

    hr {
        border-color: rgba(255,255,255,0.10);
    }


    /* Keep all important text visible on the dark background */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stText"] {
        color: white !important;
    }

    [data-testid="stDownloadButton"] button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span {
        color: white !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(90deg, #8b5cf6, #3b82f6) !important;
        color: white !important;
        border: none !important;
    }

    .student-report-card {
        background: linear-gradient(
            120deg,
            rgba(124, 58, 237, 0.32),
            rgba(37, 99, 235, 0.26)
        );
        border: 1px solid rgba(150, 160, 255, 0.35);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .student-report-name {
        color: white !important;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .student-report-registration {
        color: #dbeafe !important;
        font-size: 18px;
        font-weight: 600;
    }

    .student-report-registration span {
        color: white !important;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.html("""
        <div class="sidebar-brand">
            <div class="sidebar-icon">🎓</div>
            <div class="sidebar-title">
                Student Performance
            </div>
            <div class="sidebar-gradient">
                Predictor
            </div>
            <div class="sidebar-subtitle">
                AI Powered Academic Insights
            </div>
        </div>
        """)

    st.divider()

    if st.button("🏠  Dashboard", key="dashboard"):
        st.session_state.page = "Dashboard"

    if st.button("🔮  Prediction", key="prediction_page"):
        st.session_state.page = "Prediction"

    if st.button("📊  Analysis", key="analysis"):
        st.session_state.page = "Analysis"

    if st.button("💡  Recommendations", key="recommendations"):
        st.session_state.page = "Recommendations"

    if st.button("🎯  Target Score", key="target"):
        st.session_state.page = "Target Score"

    if st.button("📄  Reports", key="reports"):
        st.session_state.page = "Reports"

    if st.button("ℹ️  About", key="about"):
        st.session_state.page = "About"

    st.html("""
        <div class="quote-card">
            <div style="font-size:28px;">⭐</div>
            <div class="quote-text">
                Every day is a chance to improve
            </div>
            <div class="quote-sub">
                Keep learning, keep growing!
            </div>
        </div>
        """)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def create_analysis(data):

    strengths = []
    weaknesses = []
    recommendations = []

    if data["study_hours"] >= 6:
        strengths.append("Good study routine")
    else:
        weaknesses.append("Low daily study hours")
        recommendations.append(
            "Increase focused study time to at least 6 hours per day."
        )

    if data["attendance"] >= 85:
        strengths.append("Excellent attendance")
    elif data["attendance"] < 75:
        weaknesses.append("Low attendance")
        recommendations.append(
            "Improve class attendance and avoid unnecessary absences."
        )

    if data["previous_marks"] >= 75:
        strengths.append("Strong previous academic performance")
    elif data["previous_marks"] < 60:
        weaknesses.append("Previous marks require improvement")
        recommendations.append(
            "Revise weak subjects and practice previous examination questions."
        )

    if data["sleep_hours"] >= 7:
        strengths.append("Healthy sleep routine")
    else:
        weaknesses.append("Insufficient sleep")
        recommendations.append(
            "Maintain at least 7 hours of sleep each night."
        )

    if data["assignments_completed"] >= 8:
        strengths.append("Excellent assignment completion")
    elif data["assignments_completed"] < 5:
        weaknesses.append("Low assignment completion")
        recommendations.append(
            "Create an assignment schedule and complete pending work early."
        )

    if data["class_participation"] >= 7:
        strengths.append("Active classroom participation")
    elif data["class_participation"] < 5:
        weaknesses.append("Low class participation")
        recommendations.append(
            "Ask questions and actively participate in classroom discussions."
        )

    if data["screen_time"] <= 4:
        strengths.append("Controlled screen time")
    elif data["screen_time"] >= 7:
        weaknesses.append("High screen time")
        recommendations.append(
            "Reduce unnecessary screen usage during study hours."
        )

    if data["stress_level"] <= 4:
        strengths.append("Good stress management")
    elif data["stress_level"] >= 7:
        weaknesses.append("High stress level")
        recommendations.append(
            "Use planned breaks, exercise and relaxation techniques."
        )

    return strengths, weaknesses, recommendations


def get_performance(prediction):

    if prediction >= 85:
        return "Excellent", "Low Risk"

    if prediction >= 70:
        return "Good", "Low Risk"

    if prediction >= 50:
        return "Average", "Medium Risk"

    return "Needs Improvement", "High Risk"


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if st.session_state.page == "Dashboard":

    st.html("""
        <div class="page-title">
            Smart
            <span class="gradient-title">
                Student Performance
            </span>
            Predictor
        </div>

        <div class="page-subtitle">
            AI-powered academic insights for smarter learning
            and better performance.
        </div>
        """)

    st.html("""
<div class="welcome-card">
    <div class="welcome-title">👋 Welcome to your Academic Dashboard</div>
    <div class="welcome-text">
        Predict academic performance, identify strengths, discover areas
        for improvement and receive personalized recommendations using
        Machine Learning.
    </div>
</div>
""")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.html("""
            <div class="info-card">
                <div style="font-size:32px;">🤖</div>
                <div class="info-value">ML</div>
                <div class="info-label">
                    Machine Learning Prediction
                </div>
            </div>
            """)

    with col2:
        st.html("""
            <div class="info-card">
                <div style="font-size:32px;">📊</div>
                <div class="info-value">9</div>
                <div class="info-label">
                    Performance Factors
                </div>
            </div>
            """)

    with col3:
        st.html("""
            <div class="info-card">
                <div style="font-size:32px;">💡</div>
                <div class="info-value">Smart</div>
                <div class="info-label">
                    Personalized Recommendations
                </div>
            </div>
            """)

    with col4:
        st.html("""
            <div class="info-card">
                <div style="font-size:32px;">🎯</div>
                <div class="info-value">Goal</div>
                <div class="info-label">
                    Target Score Planning
                </div>
            </div>
            """)

    st.write("")

    left, right = st.columns([1.6, 1])

    with left:

        st.html("""
<div class="glass-card">
    <div class="card-title">🚀 How the System Works</div>

    <div class="recommend-box">
        1️⃣ Enter academic and lifestyle details.
    </div>

    <div class="recommend-box">
        2️⃣ The trained Random Forest model analyzes your data.
    </div>

    <div class="recommend-box">
        3️⃣ Receive a predicted final academic score.
    </div>

    <div class="recommend-box">
        4️⃣ Review strengths, weaknesses and smart recommendations.
    </div>
</div>
""")

    with right:

        if st.session_state.prediction is None:

            st.html("""
<div class="glass-card">
    <div class="card-title">📈 Latest Prediction</div>

    <div style="text-align:center; color:#aebce2; padding:35px 10px;">
        No prediction available yet.<br><br>
        Open the Prediction page to analyze a student.
    </div>
</div>
""")

        else:

            prediction = st.session_state.prediction
            performance, risk = get_performance(prediction)

            st.html(f"""
                <div class="glass-card">
                    <div class="card-title">
                        📈 Latest Prediction
                    </div>

                    <div style="text-align:center;">
                        <div class="info-value">
                            {prediction:.2f}/100
                        </div>

                        <div class="info-label">
                            {performance} • {risk}
                        </div>
                    </div>
                </div>
                """)


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif st.session_state.page == "Prediction":

    st.html("""
        <div class="page-title">
            🔮 Performance
            <span class="gradient-title">
                Prediction
            </span>
        </div>

        <div class="page-subtitle">
            Enter student information to predict the final academic score.
        </div>
        """)

    st.html('<div class="card-title">👤 Student Information</div>')

    name_col, reg_col = st.columns(2)

    with name_col:
        student_name = st.text_input(
            "Student Name",
            placeholder="Enter student name"
        )

    with reg_col:
        registration_number = st.text_input(
            "Registration Number",
            placeholder="Enter registration number"
        )

    st.write("")

    st.html('<div class="card-title">📖 Academic & Lifestyle Details</div>')

    col1, col2, col3 = st.columns(3)

    with col1:

        study_hours = st.slider(
            "🕒 Study Hours Per Day",
            1.0,
            10.0,
            5.0
        )

        sleep_hours = st.slider(
            "🛏️ Sleep Hours",
            4.0,
            10.0,
            7.0
        )

        screen_time = st.slider(
            "📱 Screen Time Per Day",
            1.0,
            10.0,
            5.0
        )

    with col2:

        attendance = st.slider(
            "✅ Attendance Percentage",
            50.0,
            100.0,
            75.0
        )

        assignments_completed = st.slider(
            "📝 Assignments Completed",
            0,
            10,
            5
        )

        extracurricular_hours = st.slider(
            "🏆 Extracurricular Hours",
            0.0,
            5.0,
            2.0
        )

    with col3:

        previous_marks = st.slider(
            "📊 Previous Marks",
            35.0,
            100.0,
            70.0
        )

        class_participation = st.slider(
            "🙋 Class Participation",
            1,
            10,
            5
        )

        stress_level = st.slider(
            "🧠 Stress Level",
            1,
            10,
            5
        )

    st.write("")

    if st.button("🔮 Predict Student Performance"):

        if not student_name.strip():
            st.error("Please enter the student name.")
            st.stop()

        if not registration_number.strip():
            st.error("Please enter the registration number.")
            st.stop()

        student = pd.DataFrame(
            [[
                study_hours,
                attendance,
                previous_marks,
                sleep_hours,
                assignments_completed,
                class_participation,
                screen_time,
                extracurricular_hours,
                stress_level
            ]],
            columns=features
        )

        prediction = model.predict(student)[0]

        student_data = {
            "student_name": student_name.strip(),
            "registration_number": registration_number.strip(),
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "sleep_hours": sleep_hours,
            "assignments_completed": assignments_completed,
            "class_participation": class_participation,
            "screen_time": screen_time,
            "extracurricular_hours": extracurricular_hours,
            "stress_level": stress_level
        }

        save_prediction(student_data, prediction)

        st.session_state.prediction = prediction
        st.session_state.student_data = student_data

        st.success("Student details and prediction saved successfully.")

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction
        performance, risk = get_performance(prediction)

        st.write("")

        result_left, result_right = st.columns([1, 2])

        with result_left:

            st.html(f"""
                <div class="glass-card">
                    <div class="card-title">
                        📊 Prediction Result
                    </div>

                    <div class="score-circle">
                        <div>
                            <div class="score-number">
                                {prediction:.2f}
                            </div>

                            <div class="score-small">
                                / 100
                            </div>
                        </div>
                    </div>
                </div>
                """)

        with result_right:

            st.html('<div class="card-title">📋 Result Summary</div>')

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "🏆 Performance Level",
                    performance
                )

            with c2:
                st.metric(
                    "🛡️ Academic Risk",
                    risk
                )

            current_marks = st.session_state.student_data["previous_marks"]

            st.metric(
                "📈 Predicted Change",
                f"{prediction - current_marks:+.2f} marks"
            )


# --------------------------------------------------
# ANALYSIS PAGE
# --------------------------------------------------

elif st.session_state.page == "Analysis":

    st.html("""
        <div class="page-title">
            📊 Student
            <span class="gradient-title">
                Analysis
            </span>
        </div>

        <div class="page-subtitle">
            Explore academic strengths and areas that require attention.
        </div>
        """)

    if st.session_state.prediction is None:

        st.warning(
            "Please complete a prediction first from the Prediction page."
        )

    else:

        data = st.session_state.student_data

        strengths, weaknesses, recommendations = create_analysis(data)

        col1, col2 = st.columns(2)

        with col1:

            st.html('<div class="card-title">💪 Student Strengths</div>')

            for strength in strengths:

                st.html(f"""
                    <div class="success-box">
                        ✅ {strength}
                    </div>
                    """)

            if not strengths:
                st.info("No major strengths detected.")

        with col2:

            st.html('<div class="card-title">⚠️ Areas for Improvement</div>')

            for weakness in weaknesses:

                st.html(f"""
                    <div class="warning-box">
                        ⚠️ {weakness}
                    </div>
                    """)

            if not weaknesses:
                st.success("No major weaknesses detected.")

        st.write("")

        st.html('<div class="card-title">📈 Performance Factor Overview</div>')

        categories = [
            "Study Hours",
            "Attendance",
            "Previous Marks",
            "Sleep",
            "Assignments",
            "Participation"
        ]

        values = [
            data["study_hours"] * 10,
            data["attendance"],
            data["previous_marks"],
            data["sleep_hours"] * 10,
            data["assignments_completed"] * 10,
            data["class_participation"] * 10
        ]

        fig, ax = plt.subplots(figsize=(10, 4))

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.bar(categories, values)

        ax.set_ylim(0, 100)

        ax.tick_params(
            axis="x",
            colors="white"
        )

        ax.tick_params(
            axis="y",
            colors="white"
        )

        ax.set_ylabel(
            "Performance Score",
            color="white"
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)


# --------------------------------------------------
# RECOMMENDATIONS PAGE
# --------------------------------------------------

elif st.session_state.page == "Recommendations":

    st.html("""
        <div class="page-title">
            💡 Personalized
            <span class="gradient-title">
                Recommendations
            </span>
        </div>

        <div class="page-subtitle">
            Smart suggestions generated from student habits and performance.
        </div>
        """)

    if st.session_state.prediction is None:

        st.warning(
            "Please complete a prediction first from the Prediction page."
        )

    else:

        data = st.session_state.student_data

        strengths, weaknesses, recommendations = create_analysis(data)

        if recommendations:

            for number, recommendation in enumerate(
                recommendations,
                start=1
            ):

                st.html(f"""
                    <div class="recommend-box">
                        💡 <b>Recommendation {number}</b><br>
                        {recommendation}
                    </div>
                    """)

        else:

            st.success(
                "Excellent academic habits. Continue maintaining your routine."
            )


# --------------------------------------------------
# TARGET SCORE PAGE
# --------------------------------------------------

elif st.session_state.page == "Target Score":

    st.html("""
        <div class="page-title">
            🎯 Target Score
            <span class="gradient-title">
                Planner
            </span>
        </div>

        <div class="page-subtitle">
            Set an academic target and understand the improvement required.
        </div>
        """)

    target_score = st.slider(
        "Choose Target Score",
        50,
        100,
        85
    )

    if st.session_state.prediction is None:

        st.info(
            "Complete a performance prediction first to compare your target."
        )

    else:

        prediction = st.session_state.prediction

        difference = target_score - prediction

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Current Predicted Score",
                f"{prediction:.2f}"
            )

        with col2:
            st.metric(
                "Target Score",
                target_score
            )

        with col3:
            st.metric(
                "Improvement Required",
                f"{max(difference, 0):.2f}"
            )

        st.write("")

        if difference <= 0:

            st.success(
                "🎉 Your predicted score already meets the selected target!"
            )

        elif difference <= 5:

            st.info(
                "You are very close to your target. Small improvements "
                "in consistency can help you reach it."
            )

        elif difference <= 15:

            st.warning(
                "Moderate improvement is required. Focus on study hours, "
                "attendance and assignment completion."
            )

        else:

            st.error(
                "Significant improvement is required. Follow a structured "
                "academic improvement plan."
            )


# --------------------------------------------------
# REPORT PAGE
# --------------------------------------------------

elif st.session_state.page == "Reports":

    st.html("""
        <div class="page-title">
            📄 Student Performance
            <span class="gradient-title">
                Report
            </span>
        </div>

        <div class="page-subtitle">
            Review and download the complete student performance report.
        </div>
        """)

    if st.session_state.prediction is None:

        st.warning(
            "Please complete a prediction before generating a report."
        )

    else:

        data = st.session_state.student_data
        prediction = st.session_state.prediction

        performance, risk = get_performance(prediction)

        strengths, weaknesses, recommendations = create_analysis(data)

        student_display_name = (
            data.get("student_name", "").strip()
            or "Student Performance Report"
        )

        registration_display = (
            data.get("registration_number", "").strip()
            or "Not provided"
        )

        st.html(
            f"""
            <div class="student-report-card">
                <div class="student-report-name">
                    👤 {student_display_name}
                </div>
                <div class="student-report-registration">
                    Registration Number:
                    <span>{registration_display}</span>
                </div>
            </div>
            """
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Predicted Score",
                f"{prediction:.2f}"
            )

        with c2:
            st.metric(
                "Performance",
                performance
            )

        with c3:
            st.metric(
                "Academic Risk",
                risk
            )

        report = f"""
SMART STUDENT PERFORMANCE REPORT

Student Name: {data["student_name"]}
Registration Number: {data["registration_number"]}

PREDICTION RESULTS

Predicted Final Score: {prediction:.2f}/100
Performance Level: {performance}
Academic Risk: {risk}

STRENGTHS

{chr(10).join("- " + item for item in strengths)}

AREAS FOR IMPROVEMENT

{chr(10).join("- " + item for item in weaknesses)}

PERSONALIZED RECOMMENDATIONS

{chr(10).join("- " + item for item in recommendations)}
"""

        st.download_button(
            "📥 Download Performance Report",
            data=report,
            file_name="student_performance_report.txt",
            mime="text/plain"
        )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif st.session_state.page == "About":

    st.html("""
        <div class="page-title">
            ℹ️ About the
            <span class="gradient-title">
                Project
            </span>
        </div>

        <div class="page-subtitle">
            Learn more about the Student Performance Prediction System.
        </div>
        """)

    st.html("""
        <div class="welcome-card">

            <div class="welcome-title">
                🎓 Smart Student Performance Predictor
            </div>

            <div class="welcome-text">

                This Machine Learning based application predicts student
                academic performance using academic and lifestyle factors.

                <br><br>

                The system analyzes study hours, attendance, previous marks,
                sleep duration, assignments, classroom participation,
                screen time, extracurricular activities and stress level.

                <br><br>

                A Random Forest Regression model is used to estimate the
                student's final academic score and provide personalized
                academic recommendations.

            </div>

        </div>
        """)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.html("""
            <div class="info-card">
                <div style="font-size:35px;">🐍</div>
                <div class="card-title">Python</div>
                <div class="info-label">
                    Core programming language
                </div>
            </div>
            """)

    with col2:

        st.html("""
            <div class="info-card">
                <div style="font-size:35px;">🤖</div>
                <div class="card-title">Machine Learning</div>
                <div class="info-label">
                    Random Forest Regression
                </div>
            </div>
            """)

    with col3:

        st.html("""
            <div class="info-card">
                <div style="font-size:35px;">⚡</div>
                <div class="card-title">Streamlit</div>
                <div class="info-label">
                    Interactive application interface
                </div>
            </div>
            """)