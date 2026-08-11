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
           APP BACKGROUND  — soft gradient with subtle pattern
        ===================================================== */

        .stApp {
            background: linear-gradient(160deg, #E8F1F8 0%, #F0F8FF 40%, #EBF5FB 70%, #F5FAFF 100%) !important;
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
            color: #334155;
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
           SIDEBAR  — Deep Navy gradient with frosted glass
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
            background: linear-gradient(135deg, #0D5C9E 0%, #1570BF 45%, #1976D2 100%) !important;
            border-radius: 24px;
            padding: 32px;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow:
                0 16px 48px rgba(13, 92, 158, 0.35),
                0 0 0 1px rgba(255,255,255,0.08) inset;
            position: relative;
            overflow: hidden;
        }

        .verdict-card::after {
            content: '';
            position: absolute;
            top: -50%; right: -20%;
            width: 220px; height: 220px;
            background: radial-gradient(circle,
                rgba(255, 255, 255, 0.12) 0%,
                transparent 70%);
            border-radius: 50%;
        }

        /* Force ALL text inside verdict card to be white/light */
        .verdict-card div,
        .verdict-card span,
        .verdict-card p {
            color: #FFFFFF !important;
        }

        .verdict-title {
            color: rgba(255,255,255,0.5) !important;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
        }

        .verdict-value {
            font-size: 28px;
            font-weight: 800;
            color: #FFFFFF !important;
            line-height: 1.2;
            position: relative;
            z-index: 1;
        }

        .metric-description-light {
            color: rgba(255,255,255,0.45) !important;
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
           BUTTONS  — Ocean Blue with glow
        ===================================================== */

        .stButton > button {
            border-radius: 16px !important;
            border: none !important;
            background: linear-gradient(135deg, #0066A1 0%, #0088CC 50%, #00A3E0 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            padding: 0.85rem 2.5rem !important;
            min-height: 56px !important;
            box-shadow:
                0 8px 24px rgba(0, 119, 182, 0.35),
                0 0 0 0 rgba(0, 180, 216, 0) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            letter-spacing: 0.3px !important;
            position: relative !important;
            overflow: hidden !important;
        }

        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow:
                0 16px 40px rgba(0, 119, 182, 0.4),
                0 0 20px rgba(0, 180, 216, 0.15) !important;
            background: linear-gradient(135deg, #005A8F 0%, #0077B6 50%, #0096C7 100%) !important;
        }

        .stButton > button:active {
            transform: translateY(-1px) !important;
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
            border-radius: 16px !important;
            background: linear-gradient(135deg, #0066A1 0%, #0088CC 50%, #00A3E0 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            min-height: 52px !important;
            box-shadow: 0 8px 24px rgba(0, 119, 182, 0.35) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            letter-spacing: 0.2px !important;
        }

        .stDownloadButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow:
                0 16px 40px rgba(0, 119, 182, 0.4),
                0 0 20px rgba(0, 180, 216, 0.15) !important;
        }

        /* Force inner text white */
        .stDownloadButton > button span,
        .stDownloadButton > button p,
        .stDownloadButton > button div {
            color: #FFFFFF !important;
        }

        /* =====================================================
           FILE UPLOADER — Frosted Glass Card
        ===================================================== */

        /* Outer Container */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.7) !important;
            border: 2px dashed rgba(0, 150, 199, 0.35) !important;
            border-radius: 20px !important;
            padding: 20px !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: rgba(0, 119, 182, 0.55) !important;
            background: rgba(255, 255, 255, 0.85) !important;
            box-shadow: 0 8px 32px rgba(0, 119, 182, 0.1) !important;
        }

        /* Inner Dropzone Area */
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(135deg, rgba(232, 244, 253, 0.8), rgba(240, 248, 255, 0.6)) !important;
            border-radius: 16px !important;
            padding: 28px !important;
            transition: background 0.3s ease !important;
        }

        /* All Dropzone Labels & Text */
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span {
            color: #334155 !important;
            font-weight: 600 !important;
        }

        /* Browse Files Button */
        [data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #0077B6, #0096C7) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            padding: 10px 22px !important;
            box-shadow: 0 6px 16px rgba(0, 119, 182, 0.25) !important;
            transition: all 0.25s ease !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 24px rgba(0, 119, 182, 0.35) !important;
        }

        [data-testid="stFileUploaderDropzone"] button p,
        [data-testid="stFileUploaderDropzone"] button span {
            color: #FFFFFF !important;
        }

        /* Uploaded File Chip */
        [data-testid="stUploadedFileData"],
        [data-testid="stFileUploaderFileData"],
        [data-testid="stFileUploaderFile"],
        [data-testid="stUploadedFile"],
        div[data-testid="stFileUploader"] > section > div,
        div[data-testid="stFileUploaderFile"] {
            background: linear-gradient(135deg, #DBEAFE, #E0F2FE) !important;
            border: 1px solid #93C5FD !important;
            border-radius: 14px !important;
            padding: 10px 14px !important;
        }

        /* Text & Details Inside Uploaded File Chip */
        [data-testid="stUploadedFileName"],
        [data-testid="stFileUploaderFileName"],
        [data-testid="stUploadedFileData"] span,
        [data-testid="stUploadedFileData"] div,
        [data-testid="stUploadedFileData"] small,
        [data-testid="stFileUploaderFile"] span,
        [data-testid="stFileUploaderFile"] div,
        [data-testid="stFileUploaderFile"] small {
            background-color: transparent !important;
            color: #0A1628 !important;
            font-weight: 700 !important;
        }

        /* Delete Icon Button */
        [data-testid="stFileUploaderDeleteBtn"],
        [data-testid="stFileUploaderFileDeleteBtn"],
        [data-testid="stUploadedFileData"] button,
        [data-testid="stFileUploaderFile"] button {
            background: #E0F2FE !important;
            border: 1px solid #BAE6FD !important;
            border-radius: 8px !important;
        }

        [data-testid="stUploadedFileData"] svg,
        [data-testid="stFileUploaderFile"] svg {
            fill: #0077B6 !important;
            color: #0077B6 !important;
        }

        /* =====================================================
           TEXT AREA — Elevated with smooth focus
        ===================================================== */

        .stTextArea textarea {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border-radius: 16px !important;
            border: 2px solid #E2E8F0 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            line-height: 1.7 !important;
            padding: 18px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
        }

        .stTextArea textarea:hover {
            border-color: #94A3B8 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06) !important;
        }

        .stTextArea textarea:focus {
            border-color: #0077B6 !important;
            box-shadow:
                0 0 0 4px rgba(0, 119, 182, 0.12),
                0 4px 20px rgba(0, 119, 182, 0.08) !important;
            outline: none !important;
        }

        .stTextArea textarea::placeholder {
            color: #94A3B8 !important;
            font-style: italic !important;
            font-weight: 400 !important;
        }

        .stTextArea label, .stTextArea label p {
            color: #1E293B !important;
            font-weight: 600 !important;
        }

        /* =====================================================
           DIVIDER — Refined gradient
        ===================================================== */

        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(0, 119, 182, 0.12) 20%,
                rgba(0, 119, 182, 0.2) 50%,
                rgba(0, 119, 182, 0.12) 80%,
                transparent 100%) !important;
            margin: 32px 0 !important;
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
           ANIMATIONS
        ===================================================== */

        @keyframes pulse-dot {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
            50%       { box-shadow: 0 0 0 7px rgba(16,185,129,0); }
        }

        .pulse-dot {
            animation: pulse-dot 2s infinite;
        }

        @keyframes shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-8px); }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def load_css():
    apply_styles()