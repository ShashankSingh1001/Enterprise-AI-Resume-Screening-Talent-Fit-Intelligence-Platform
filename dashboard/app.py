"""
Resume AI Platform - Streamlit Dashboard
Minimal version for Phase 1 testing
"""

import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="Resume AI Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚀 Resume AI Platform")
st.markdown("### Enterprise AI Resume Screening & Talent Fit Intelligence")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    st.info("Dashboard is under development")
    st.markdown("---")
    st.markdown("**Phase 1: Setup Complete** ✅")
    st.markdown("**Current Phase:** Building Infrastructure")

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Project Status",
        value="Phase 1",
        delta="Setup Complete"
    )

with col2:
    st.metric(
        label="Environment",
        value=os.getenv("ENVIRONMENT", "development").upper()
    )

with col3:
    st.metric(
        label="Version",
        value="1.0.0"
    )

st.markdown("---")

# Status section
st.subheader("📊 System Status")

status_data = {
    "Component": ["API", "Database", "MLflow", "Dashboard"],
    "Status": ["🟢 Running", "🟢 Connected", "🟢 Tracking", "🟢 Active"],
    "Port": ["8000", "5432", "5000", "8501"]
}

st.table(status_data)

st.markdown("---")

# Information
st.subheader("ℹ️ Next Steps")
st.markdown("""
This is a minimal dashboard for Phase 1 testing. Full features will be added in Phase 11.

**Completed:**
- ✅ Project Structure
- ✅ Environment Setup
- ✅ Docker Configuration
- ✅ Database Schema
- ✅ MLflow Setup

**Next Phase:**
- 🔄 Core Infrastructure & Utilities
- 🔄 Logging System
- 🔄 Custom Exceptions
- 🔄 Utility Functions
""")

st.success("Phase 1 Setup Complete! Ready for development.")