import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain

from app.arxiv_handler import fetch_arxiv_pdfs
from app.ingestion.parser import extract_text_from_pdf
from app.ingestion.splitter import split_text
from app.retrieval.embedder import get_embedder
from app.retrieval.retriever import build_vectorstore

# Load environment variables
load_dotenv()

def configure_page():
    """Configure Streamlit page settings and styling"""
    st.set_page_config(
        page_title="Doc Q&A AI Agent", 
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .paper-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .answer-box {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #ecf0f1;
    }
    .source-box {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
        color: #ecf0f1;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'docs' not in st.session_state:
        st.session_state.docs = []
        st.session_state.titles = []
        st.session_state.paper_count = 0

def render_header():
    """Render the main header section"""
    st.markdown('<h1 class="main-header">🤖 Research Paper Q&A Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🔍 Discover, analyze, and interact with arXiv research papers using AI</p>', unsafe_allow_html=True)

def render_search_section():
    """Render the paper search and fetch section"""
    st.markdown("## 📚 Paper Discovery")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter research topic or keywords:",
            placeholder="e.g., machine learning, neural networks, quantum computing...",
            help="Search for relevant research papers on arXiv"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        fetch_button = st.button("🚀 Fetch Papers", type="primary", use_container_width=True)
    
    return query, fetch_button

def fetch_and_process_papers(query):
    """Fetch and process papers from arXiv"""
    if not query.strip():
        st.warning("⚠️ Please enter a valid research topic")
        return False
    
    with st.spinner("🔄 Searching arXiv database..."):
        try:
            results = fetch_arxiv_pdfs(query, max_results=3)
            
            if not results:
                st.error("❌ No papers found for this topic")
                return False
            
            # Process papers
            progress_bar = st.progress(0)
            for i, (pdf_path, title, paper_id) in enumerate(results):
                raw_text = extract_text_from_pdf(pdf_path)
                chunks = split_text(raw_text, metadata={"source": title})
                st.session_state.docs.extend(chunks)
                st.session_state.titles.append(title)
                progress_bar.progress((i + 1) / len(results))
            
            st.session_state.paper_count = len(results)
            st.success(f"✅ Successfully loaded {len(results)} papers with {len(st.session_state.docs)} text chunks")
            return True
            
        except Exception as e:
            st.error(f"❌ Error fetching papers: {str(e)}")
            return False

def render_paper_summary():
    """Display loaded papers summary"""
    if st.session_state.titles:
        st.markdown("## 📄 Loaded Research Papers")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1f77b4; margin: 0;">📊 {len(st.session_state.titles)}</h3>
                <p style="margin: 0;">Papers Loaded</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #28a745; margin: 0;">📝 {len(st.session_state.docs)}</h3>
                <p style="margin: 0;">Text Chunks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #dc3545; margin: 0;">🔍 Ready</h3>
                <p style="margin: 0;">For Questions</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display paper titles in cards
        for i, title in enumerate(st.session_state.titles):
            st.markdown(f"""
            <div class="paper-card">
                <strong>Paper {i+1}:</strong> {title}
            </div>
            """, unsafe_allow_html=True)

def render_qa_section():
    """Render the Q&A interface"""
    if not st.session_state.titles:
        st.info("📋 Load some research papers first to start asking questions")
        return "", False
    
    st.markdown("## 💭 Ask Questions")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_area(
            "What would you like to know?",
            placeholder="e.g., What are the main findings? How do these papers compare? What methodologies are used?",
            height=100,
            help="Ask questions about the loaded research papers"
        )
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing
        ask_button = st.button("🧠 Get Answer", type="primary", use_container_width=True)
    
    return question, ask_button

def process_question(question):
    """Process the user's question and generate answer"""
    if not question.strip():
        st.warning("⚠️ Please enter a question")
        return
    
    with st.spinner("🤔 Analyzing papers and generating answer..."):
        try:
            # Build retrieval system
            embedder = get_embedder()
            vectorstore = build_vectorstore(st.session_state.docs, embedder)
            retriever = vectorstore.as_retriever()
            
            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Create QA chain
            qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm, 
                retriever=retriever, 
                chain_type="stuff"
            )
            
            # Get answer
            result = qa_chain({"question": question})
            display_answer(result["answer"], result["sources"])
            
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")

def display_answer(answer, sources):
    """Display the AI-generated answer and sources"""
    st.markdown("### 🎯 AI Response")
    st.markdown(f"""
    <div class="answer-box">
        {answer}
    </div>
    """, unsafe_allow_html=True)
    
    if sources and sources.strip():
        st.markdown("### 📚 Sources Referenced")
        source_list = [s.strip() for s in sources.split("\n") if s.strip()]
        
        for i, source in enumerate(source_list):
            st.markdown(f"""
            <div class="source-box">
                <strong>Source {i+1}:</strong> {source}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No specific sources were identified for this answer")

def render_sidebar():
    """Render sidebar with additional information and controls"""
    with st.sidebar:
        st.markdown("## 🔧 Controls")
        
        if st.button("🗑️ Clear All Papers", use_container_width=True):
            st.session_state.docs = []
            st.session_state.titles = []
            st.session_state.paper_count = 0
            st.rerun()
        
        st.markdown("---")
        st.markdown("## ℹ️ How it Works")
        st.markdown("""
        1. **Search**: Enter keywords to find relevant papers
        2. **Fetch**: Download and process papers from arXiv
        3. **Ask**: Query the AI about paper contents
        4. **Learn**: Get insights across multiple papers
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 Tips")
        st.markdown("""
        - Use specific keywords for better results
        - Ask comparative questions across papers
        - Request summaries of key findings
        - Inquire about methodologies used
        """)

def main():
    """Main application function"""
    configure_page()
    initialize_session_state()
    render_sidebar()
    
    # Main content area
    render_header()
    
    # Paper search section
    query, fetch_button = render_search_section()
    
    if fetch_button:
        fetch_and_process_papers(query)
    
    # Show loaded papers
    render_paper_summary()
    
    st.markdown("---")
    
    # Q&A section
    question, ask_button = render_qa_section()
    
    if ask_button and question:
        process_question(question)

if __name__ == "__main__":
    main()