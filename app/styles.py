"""
styles.py

Apple UI Theme
"""

import streamlit as st


def load_css():

    st.markdown("""
<style>
/* Main text */
html, body, .stApp {
    color: #1D1D1F;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #1D1D1F !important;
}

/* Streamlit markdown */
.stMarkdown {
    color: #1D1D1F;
}

/* Labels */
label {

    color: #1D1D1F !important;

    font-weight: 600;

}

/* Expander header */
.streamlit-expanderHeader {
    color: #1D1D1F !important;
}

/* Metrics */
[data-testid="stMetricLabel"] {
    color: #6E6E73 !important;
}

[data-testid="stMetricValue"] {
    color: #1D1D1F !important;
}

/* Background */

.stApp{

    background:#F5F5F7;

}

/* Main Title */

.main-title{

    font-size:46px;

    font-weight:700;

    text-align:center;

    color:#1D1D1F;

}

/* Subtitle */

.sub-title{

    text-align:center;

    color:#6E6E73;

    font-size:20px;

    margin-bottom:30px;

}

/* Glass Card */

.glass{

    background:rgba(255,255,255,.80);

    backdrop-filter:blur(25px);

    border-radius:24px;

    padding:25px;

    box-shadow:
        0 8px 25px rgba(0,0,0,.08);

}

/* Button */

.stButton > button {

    width:100%;

    background:#007AFF;

    color:white !important;

    font-weight:700;

    border-radius:16px;

    border:none;

    padding:14px;

}

.stDownloadButton > button {

    width:100%;

    background:#007AFF !important;

    color:white !important;

    font-weight:700;

    border-radius:16px;

    border:none;

    padding:14px;

}

/* Skill Chips */

.skill-chip{

    display:inline-block;

    padding:10px 18px;

    margin:5px;

    border-radius:999px;

    color:white;

    font-weight:600;

}

.skill-match{

    background:#007AFF;

}

.skill-missing{

    background:#FF453A;

}

.metric-card{

    background:white;

    border-radius:20px;

    padding:20px;

    text-align:center;

    box-shadow:
        0 10px 25px rgba(0,0,0,.08);

}

.metric-title{

    color:#6E6E73;

    font-size:18px;

}

.metric-value{

    font-size:40px;

    font-weight:700;

}
/* ===========================
/* ==========================================
   TEXT AREA (New Streamlit)
========================================== */

[data-testid="stTextArea"] textarea{

    background:#FFFFFF !important;

    color:#1D1D1F !important;

    caret-color:#1D1D1F !important;

    border:1px solid #D2D2D7 !important;

    border-radius:18px !important;

    font-size:16px !important;

}

[data-testid="stTextArea"] textarea::placeholder{

    color:#8E8E93 !important;

}

[data-testid="stTextArea"] textarea:focus{

    border:2px solid #007AFF !important;

    box-shadow:0 0 0 3px rgba(0,122,255,.15) !important;

}

/* ==========================================
   FILE UPLOADER
========================================== */

[data-testid="stFileUploader"]{

    background:#FFFFFF !important;

    border:2px dashed #007AFF !important;

    border-radius:18px !important;

    padding:18px !important;

}

[data-testid="stFileUploaderDropzone"]{

    background:#FFFFFF !important;

    border:none !important;

}

[data-testid="stFileUploader"] small{

    color:#6E6E73 !important;

}

/* ==========================================
   INPUT LABELS
========================================== */

[data-testid="stFileUploader"] label,
[data-testid="stTextArea"] label{

    color:#1D1D1F !important;

    font-weight:600 !important;

}

/* ==========================================
   SIDEBAR
========================================== */

section[data-testid="stSidebar"]{

    background:#FFFFFF !important;

}
</style>
""", unsafe_allow_html=True)