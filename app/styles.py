"""
styles.py

Professional Premium UI — Deep Navy Blue / Cerulean theme
"""

import streamlit as st


def apply_styles():

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* =====================================================
           COLOR TOKENS
           Primary  : #0077B6  (Ocean Blue)
           Accent   : #00B4D8  (Cerulean)
           Dark     : #0A1628  (Deep Navy)
           BG       : #F0F8FF  (Alice Blue)
        ===================================================== */

        /* =====================================================
           GLOBAL RESET
        ===================================================== */

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        /* =====================================================
           APP BACKGROUND  — soft alice-blue wash
        ===================================================== */

        .stApp {
            background: linear-gradient(160deg, #EBF5FB 0%, #F0F8FF 55%, #E8F4FD 100%) !important;
            min-height: 100vh;
        }

        .block-container {
            max-width: 1400px !important;
            padding: 1rem 2.5rem 4rem 2.5rem !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: none !important;
        }

        /* =====================================================
           FORCE DARK TEXT EVERYWHERE IN MAIN CONTENT
           Fixes the invisible-text bug
        ===================================================== */

        section[data-testid="stMain"] p,
        section[data-testid="stMain"] span,
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] li,
        section[data-testid="stMain"] div,
        .stMarkdown p,
        .stMarkdown span,
        .stMarkdown li {
            color: #0F172A;
        }

        /* Streamlit bordered container text */
        [data-testid="stVerticalBlockBorderWrapper"] p,
        [data-testid="stVerticalBlockBorderWrapper"] span,
        [data-testid="stVerticalBlockBorderWrapper"] label,
        [data-testid="stVerticalBlockBorderWrapper"] div {
            color: #1E293B !important;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #0A1628 !important;
            letter-spacing: -0.5px !important;
        }

        /* Caption text */
        [data-testid="stCaptionContainer"] p,
        small {
            color: #64748B !important;
        }

        /* Metric labels & values */
        [data-testid="stMetricLabel"] label,
        [data-testid="stMetricLabel"] p {
            color: #475569 !important;
        }
        [data-testid="stMetricValue"] div {
            color: #0A1628 !important;
        }

        /* =====================================================
           SIDEBAR  — Deep Navy gradient
        ===================================================== */

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071B2F 0%, #0A2744 55%, #071B2F 100%) !important;
            border-right: none !important;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label {
            color: #CBD5E1 !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #E2E8F0 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.08) !important;
        }

        [data-testid="stSidebar"] .stSuccess > div {
            background: rgba(16, 185, 129, 0.14) !important;
            border: 1px solid rgba(16, 185, 129, 0.3) !important;
            color: #6EE7B7 !important;
            border-radius: 10px !important;
        }

        [data-testid="stSidebar"] .stSuccess p {
            color: #6EE7B7 !important;
        }

        [data-testid="stSidebar"] .stInfo > div {
            background: rgba(0,119,182,0.18) !important;
            border: 1px solid rgba(0,180,216,0.35) !important;
            border-radius: 10px !important;
        }

        [data-testid="stSidebar"] .stInfo p {
            color: #7DD3F7 !important;
        }

        /* =====================================================
           METRIC CARDS
        ===================================================== */

        .metric-card {
            background: #FFFFFF;
            border: 1px solid rgba(0, 119, 182, 0.12);
            border-radius: 20px;
            padding: 22px 24px;
            min-height: 130px;
            box-shadow:
                0 4px 20px rgba(0, 119, 182, 0.08),
                0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.25s cubic-bezier(0.4,0,0.2,1),
                        box-shadow 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #0077B6, #00B4D8);
            border-radius: 20px 20px 0 0;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow:
                0 14px 36px rgba(0, 119, 182, 0.15),
                0 4px 8px rgba(0,0,0,0.06);
        }

        .metric-label {
            color: #475569;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #0A1628;
            font-size: 30px;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -1px;
        }

        .metric-description {
            color: #64748B;
            font-size: 12px;
            margin-top: 6px;
            font-weight: 500;
        }

        /* =====================================================
           SCORE CARD
        ===================================================== */

        .score-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #EBF5FB 100%);
            border: 1px solid rgba(0, 119, 182, 0.15);
            border-radius: 24px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0, 119, 182, 0.12);
        }

        .score-label {
            color: #0077B6;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        /* =====================================================
           VERDICT CARD  — Deep Navy gradient
        ===================================================== */

        .verdict-card {
            background: linear-gradient(135deg, #0E4C92 0%, #1E90FF 50%, #4facfe 100%);   
            border-radius: 24px;
            padding: 32px;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 20px 60px rgba(0, 77, 130, 0.35);
            position: relative;
            overflow: hidden;
        }

        .verdict-card::after {
            content: '';
            position: absolute;
            top: -50%; right: -20%;
            width: 220px; height: 220px;
            background: radial-gradient(circle,
                rgba(0, 180, 216, 0.35) 0%,
                transparent 70%);
            border-radius: 50%;
        }

        .verdict-title {
            color: rgba(255,255,255,0.5);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
        }

        .verdict-value {
            font-size: 26px;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.2;
            position: relative;
            z-index: 1;
        }

        .metric-description-light {
            color: rgba(255,255,255,0.5);
            font-size: 13px;
            margin-top: 12px;
            position: relative;
            z-index: 1;
            font-weight: 500;
        }

        /* =====================================================
           SKILLS
        ===================================================== */

        .skills-container {
            background: #FFFFFF;
            border: 1px solid #DBEAFE;
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        }

        .skill-chip {
            display: inline-block;
            padding: 7px 14px;
            margin: 4px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
        }

        .skill-match {
            background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
            color: #065F46;
            border: 1px solid #A7F3D0;
        }

        .skill-missing {
            background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
            color: #991B1B;
            border: 1px solid #FECACA;
        }

        /* =====================================================
           ANALYTICS CARD
        ===================================================== */

        .analytics-card {
            background: #FFFFFF;
            border: 1px solid #DBEAFE;
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        }

        /* =====================================================
           BUTTONS  — Ocean Blue
        ===================================================== */

        .stButton > button {
            border-radius: 14px !important;
            border: none !important;
            background: linear-gradient(135deg, #0077B6, #0096C7) !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            padding: 0.75rem 2rem !important;
            box-shadow: 0 8px 24px rgba(0, 119, 182, 0.35) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 14px 32px rgba(0, 119, 182, 0.45) !important;
        }

        /* Force inner text white */
        .stButton > button span,
        .stButton > button p,
        .stButton > button div {
            color: #FFFFFF !important;
        }

        /* =====================================================
           DOWNLOAD BUTTON
        ===================================================== */

        .stDownloadButton > button {
            border-radius: 14px !important;
            background: linear-gradient(135deg, #0077B6, #0096C7) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            min-height: 52px !important;
            box-shadow: 0 8px 24px rgba(0, 119, 182, 0.35) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            letter-spacing: 0.2px !important;
        }

        .stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 14px 32px rgba(0, 119, 182, 0.45) !important;
        }

        /* Force inner text white */
        .stDownloadButton > button span,
        .stDownloadButton > button p,
        .stDownloadButton > button div {
            color: #FFFFFF !important;
        }

        /* =====================================================
           FILE UPLOADER
        ===================================================== */

        [data-testid="stFileUploader"] {
            background: #FFFFFF !important;
            border: 2px dashed #BAE6FD !important;
            border-radius: 18px !important;
            padding: 14px !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #F0F8FF !important;
            border-radius: 14px !important;
            padding: 24px !important;
        }

        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span {
            color: #475569 !important;
            font-weight: 500 !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #EBF5FB, #DBEAFE) !important;
            color: #0077B6 !important;
            border: 1px solid #BAE6FD !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            padding: 6px 16px !important;
            box-shadow: none !important;
        }

        /* Uploaded File Container & Card Fix */
        [data-testid="stUploadedFileData"],
        [data-testid="stFileUploaderFileData"],
        [data-testid="stFileUploaderFile"],
        [data-testid="stUploadedFile"],
        div[data-testid="stFileUploader"] > section > div {
            background: #F0F8FF !important;
            border: 1px solid #BAE6FD !important;
            border-radius: 12px !important;
            padding: 10px 14px !important;
            color: #0A1628 !important;
        }

        [data-testid="stUploadedFileName"],
        [data-testid="stFileUploaderFileName"],
        [data-testid="stUploadedFileData"] span,
        [data-testid="stUploadedFileData"] div,
        [data-testid="stFileUploaderFile"] span,
        [data-testid="stFileUploaderFile"] div {
            background-color: #E0F2FE !important; 
            font-weight: 600 !important;
        }

        [data-testid="stFileUploaderDeleteBtn"],
        [data-testid="stFileUploaderFileDeleteBtn"],
        [data-testid="stUploadedFileData"] button,
        [data-testid="stFileUploaderFile"] button {
            background-color: #FFFFFF !important;  
            background: transparent !important;
            border: none !important;
        }

        [data-testid="stUploadedFileData"] svg,
        [data-testid="stFileUploaderFile"] svg {
            fill: #0077B6 !important;
            color: #0077B6 !important;
        }

        /* =====================================================
           TEXT AREA
        ===================================================== */

        .stTextArea textarea {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border-radius: 14px !important;
            border: 2px solid #DBEAFE !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            line-height: 1.6 !important;
            padding: 14px !important;
        }

        .stTextArea textarea:focus {
            border-color: #0077B6 !important;
            box-shadow: 0 0 0 3px rgba(0, 119, 182, 0.15) !important;
            outline: none !important;
        }

        .stTextArea textarea::placeholder {
            color: #94A3B8 !important;
            font-style: italic !important;
        }

        .stTextArea label, .stTextArea label p {
            color: #1E293B !important;
            font-weight: 600 !important;
        }

        /* =====================================================
           DIVIDER
        ===================================================== */

        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, #CBD5E1, transparent) !important;
            margin: 28px 0 !important;
        }

        /* =====================================================
           PROGRESS BAR  — Ocean Blue
        ===================================================== */

        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #0077B6, #00B4D8) !important;
            border-radius: 999px !important;
        }

        [data-testid="stProgress"] > div {
            border-radius: 999px !important;
            background: #DBEAFE !important;
        }

        /* =====================================================
           EXPANDER
        ===================================================== */

        .streamlit-expanderHeader {
            border-radius: 14px !important;
            font-weight: 600 !important;
            background: #F8FBFF !important;
        }

        /* =====================================================
           ALERT / STATUS BOXES
        =====================================================  */

        .stSuccess > div {
            background: #ECFDF5 !important;
            border: 1px solid #A7F3D0 !important;
            border-radius: 12px !important;
        }
        .stSuccess p { color: #065F46 !important; }

        .stError > div {
            background: #FEF2F2 !important;
            border: 1px solid #FECACA !important;
            border-radius: 12px !important;
        }
        .stError p { color: #991B1B !important; }

        .stWarning > div {
            background: #FFFBEB !important;
            border: 1px solid #FDE68A !important;
            border-radius: 12px !important;
        }
        .stWarning p { color: #92400E !important; }

        .stInfo > div {
            background: #EFF6FF !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 12px !important;
        }
        .stInfo p { color: #1E40AF !important; }

        /* =====================================================
           NATIVE STREAMLIT CONTAINER (border=True)
           Force readable text inside bordered containers
        ===================================================== */

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border-radius: 16px !important;
            border-color: #DBEAFE !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] p,
        [data-testid="stVerticalBlockBorderWrapper"] span,
        [data-testid="stVerticalBlockBorderWrapper"] div {
            color: #1E293B !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] h3,
        [data-testid="stVerticalBlockBorderWrapper"] h2 {
            color: #0A1628 !important;
        }

        /* =====================================================
           PULSE ANIMATION (header dot)
        ===================================================== */

        @keyframes pulse-dot {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
            50%       { box-shadow: 0 0 0 7px rgba(16,185,129,0); }
        }

        .pulse-dot {
            animation: pulse-dot 2s infinite;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def load_css():
    apply_styles()