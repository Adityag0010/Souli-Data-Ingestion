"""
Main Streamlit app with multi-page navigation
"""
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Souli Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🎯 Souli Data Ingestion & Chatbot")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Select Interface",
        [
            "🥇 Gold Data Viewer",
            "🚀 Data Ingestion (Improved)",
            "🧬 Multi-Collection Ingestion",
            "🎬 Data Ingestion",
            "💬 Chatbot Testing",
            "🔬 Dev Testing",
        ],
        help="Choose between data ingestion or chatbot testing"
    )

    if page == "🥇 Gold Data Viewer":
        from pages import gold_viewer
        gold_viewer.show()
    elif page == "🚀 Data Ingestion (Improved)":
        from pages import data_ingestion_improved
        data_ingestion_improved.show()
    elif page == "🎬 Data Ingestion":
        from pages import data_ingestion
        data_ingestion.show()
    elif page == "💬 Chatbot Testing":
        from pages import chatbot_testing
        chatbot_testing.show()
    elif page == "🔬 Dev Testing":
        from pages import dev_testing
        dev_testing.show()
    elif page == "🧬 Multi-Collection Ingestion":
        from pages import multi_data_ingestion_improved
        multi_data_ingestion_improved.show()

if __name__ == "__main__":
    main()