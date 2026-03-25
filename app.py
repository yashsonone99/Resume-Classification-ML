import streamlit as st
import pickle
import pandas as pd
import pdfplumber
from docx import Document
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="Resume Classification System", layout="wide")

# ---------------- BACKGROUND ----------------
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background: rgba(0,0,0,0.65);
            padding: 2rem;
            border-radius: 10px;
        }}

        /* DOWNLOAD BUTTON STYLE (same as predict button) */
        div.stDownloadButton > button {{
            width:100%;
            background-color:transparent;
            color:#ff4b4b;
            border:1px solid #ff4b4b;
            font-size:16px;
            border-radius:8px;
            padding:10px;
            font-weight:500;
        }}

        div.stDownloadButton > button:hover {{
            background-color:#ff4b4b;
            color:white;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.png")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("resume_model.pkl","rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align:center; font-size:48px; margin-bottom:10px;'>
    Resume Classification System
    </h1>

    <p style='text-align:center; font-size:18px; color:#cccccc;'>
    Upload multiple resumes to classify them automatically.
    </p>

    <hr style="
        width:60%;
        margin:auto;
        border:1px solid #4cc9f0;
        margin-top:10px;
        margin-bottom:20px;
    ">
    """,
    unsafe_allow_html=True
)

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):

    if file.name.endswith(".pdf"):
        text=""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)

    return ""

# ---------------- TOP LAYOUT ----------------
upload_col, chart_col = st.columns([1,1])

with upload_col:

    st.subheader("Upload Resumes")

    uploaded_files = st.file_uploader(
        "Drag and drop resumes here",
        accept_multiple_files=True
    )

    predict = st.button("Predict Job Roles")

with chart_col:
    chart_area = st.empty()

# ---------------- PROCESS ----------------
if predict and uploaded_files:

    results=[]
    previews={}

    for file in uploaded_files:

        text = extract_text(file)

        vector = tfidf.transform([text])
        prediction = model.predict(vector)

        role = le.inverse_transform(prediction)[0]

        preview = text[:150]

        results.append({
            "File Name":file.name,
            "Predicted Role":role,
            "Preview":preview
        })

        previews[file.name] = text

    df = pd.DataFrame(results)

    # ---------------- VISUALIZATION ----------------
    
    with chart_area:

        st.subheader("Category Distribution")

        role_counts = df["Predicted Role"].value_counts()

        fig, ax = plt.subplots(figsize=(2,1), dpi=200)

        colors = ["#00c2ff","#4cc9f0","#7209b7","#f72585","#4361ee"]

        ax.bar(role_counts.index, role_counts.values, color=colors)

        fig.patch.set_alpha(0)

        # CLEAN OUTER BORDER (theme matching)
        fig.patch.set_edgecolor("#4cc9f0")
        fig.patch.set_linewidth(2)

        ax.set_facecolor("none")

        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(1)

        ax.set_xlabel("Role", color="white", fontsize=5)
        ax.set_ylabel("Resumes", color="white", fontsize=5)

        # -------- DYNAMIC Y AXIS --------
        max_val = role_counts.values.max()
        step = max(1, max_val // 5)
        ax.set_ylim(0, max_val + step)
        ax.set_yticks(range(0, max_val + step + 1, step))

        ax.tick_params(colors="white", labelsize=4)

        plt.xticks(rotation=25, ha="right")

        for i, v in enumerate(role_counts.values):
            ax.text(i, v + 0.1, str(v), ha="center", color="white", fontsize=5)

        st.pyplot(fig)

    # ---------------- TABLE ----------------
    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    # ---------------- SORTED + PREVIEW ----------------
    sorted_col, preview_col = st.columns([1,1])

    with sorted_col:

        st.subheader("Sorted Resumes")

        categories = df.groupby("Predicted Role")

        for role, data in categories:

            st.markdown(f"### {role}")

            for file in data["File Name"]:

                st.markdown(
                    f"""
                    <div style="
                        background:#1f2933;
                        padding:10px;
                        border-radius:8px;
                        margin-bottom:6px;
                        border-left:4px solid #4cc9f0;
                    ">
                    {file}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with preview_col:

        st.subheader("Resume Preview")

        selected = st.selectbox(
            "Select Resume",
            list(previews.keys())
        )

        st.text_area(
            "",
            previews[selected][:800],
            height=320
        )

    # ---------------- DOWNLOAD BUTTON ----------------

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Resume Classification Report",
        csv,
        "resume_classification_report.csv",
        "text/csv"
    )