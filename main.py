import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import traceback
import re
from app.arxiv_handler import fetch_arxiv_pdfs
from app.ingestion.parser import extract_text_from_pdf
from app.ingestion.splitter import split_text
from app.retrieval.embedder import get_embedder
from app.retrieval.retriever import build_vectorstore

# Load environment variables
load_dotenv()

def configure_app():
    """Configure Streamlit app settings and custom styling"""
    st.set_page_config(
        page_title="AI Research Assistant", 
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #6C757D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .paper-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    .assistant-message {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #ecf0f1;
        margin-right: 2rem;
    }
    .source-item {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #ecf0f1;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #28a745;
    }
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .continue-prompt {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }
    .error-container {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session():
    """Initialize all session state variables"""
    session_vars = {
        'docs': [],
        'titles': [],
        'chat_history': [],
        'vectorstore': None,
        'qa_chain': None,
        'conversation_active': False,
        'papers_loaded': False,
        'last_error': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['GOOGLE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please set these in your .env file or environment")
        return False
    return True

def render_header():
    """Render the main application header"""
    st.markdown('<h1 class="main-title">🤖 Doc Q&A AI Agent V-2</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent conversations with research papers • Powered by LangChain + Gemini + ArXiv</p>', unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with controls and information"""
    with st.sidebar:
        st.markdown("## 🎛️ Controls")
        
        # Reset conversation
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_active = False
            st.session_state.last_error = None
            st.rerun()
        
        # Clear all data
        if st.button("🗑️ Clear All Data", use_container_width=True):
            for key in ['docs', 'titles', 'chat_history', 'vectorstore', 'qa_chain']:
                st.session_state[key] = [] if key in ['docs', 'titles', 'chat_history'] else None
            st.session_state.conversation_active = False
            st.session_state.papers_loaded = False
            st.session_state.last_error = None
            st.rerun()
        
        st.markdown("---")
        
        # Status information
        st.markdown("## 📊 Status")
        papers_count = len(st.session_state.titles)
        chunks_count = len(st.session_state.docs)
        
        st.markdown(f"""
        <div class="status-card">
            <h4 style="color: #2E86AB; margin: 0;">📚 {papers_count}</h4>
            <p style="margin: 0.2rem 0;">Papers Loaded</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="status-card">
            <h4 style="color: #28a745; margin: 0;">📝 {chunks_count}</h4>
            <p style="margin: 0.2rem 0;">Text Chunks</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Error display
        if st.session_state.last_error:
            st.markdown("## ⚠️ Last Error")
            st.error(st.session_state.last_error)
            if st.button("Clear Error", use_container_width=True):
                st.session_state.last_error = None
                st.rerun()
        
        st.markdown("---")
        
        # Usage guide
        st.markdown("## 💡 How to Use")
        st.markdown("""
        1. **Search** for research papers by topic
        2. **Load** papers from arXiv database
        3. **Ask** questions about the content
        4. **Continue** the conversation naturally
        5. **Start new** conversation anytime
        """)

def render_search_section():
    """Render the paper search interface"""
    st.markdown("## 🔍 Paper Discovery")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Search Topic:",
            placeholder="Enter research topic (e.g., machine learning, quantum computing, neural networks...)",
            help="Search for relevant papers on arXiv"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🚀 Search & Load", type="primary", use_container_width=True)
    
    return query, search_button

def fetch_papers(query):
    """Fetch and process papers from arXiv with better error handling"""
    if not query.strip():
        st.warning("⚠️ Please enter a research topic to search")
        return False
    
    # Clear previous error
    st.session_state.last_error = None
    
    try:
        with st.spinner("🔄 Searching arXiv database..."):
            st.write(f"📡 Final Query: {query}")  # update sr-1
            results = fetch_arxiv_pdfs(query, max_results=3)
            
            if not results:
                error_msg = "❌ No papers found for this topic. Try different keywords."
                st.error(error_msg)
                st.session_state.last_error = error_msg
                return False
            
            st.success(f"✅ Found {len(results)} papers. Processing...")
            
        # Clear previous data
        st.session_state.docs = []
        st.session_state.titles = []
        
        # Process each paper with individual error handling
        with st.spinner("📖 Processing papers..."):
            progress_bar = st.progress(0)
            processed_count = 0
            
            for i, (pdf_path, title, paper_id) in enumerate(results):
                try:
                    # Check if file exists
                    if not os.path.exists(pdf_path):
                        st.warning(f"⚠️ PDF file not found: {title}")
                        continue
                        
                    raw_text = extract_text_from_pdf(pdf_path)
                    
                    if not raw_text or len(raw_text.strip()) < 100:
                        st.warning(f"⚠️ Could not extract meaningful text from: {title}")
                        continue
                    
                    chunks = split_text(raw_text, metadata={"source": title, "paper_id": paper_id})
                    
                    if chunks:
                        st.session_state.docs.extend(chunks)
                        st.session_state.titles.append(title)
                        processed_count += 1
                        st.info(f"✅ Processed: {title} ({len(chunks)} chunks)")
                    else:
                        st.warning(f"⚠️ No text chunks created for: {title}")
                        
                except Exception as e:
                    error_msg = f"Error processing {title}: {str(e)}"
                    st.warning(f"⚠️ {error_msg}")
                    st.session_state.last_error = error_msg
                    continue
                finally:
                    progress_bar.progress((i + 1) / len(results))
            
            if processed_count == 0:
                error_msg = "❌ Failed to process any papers successfully"
                st.error(error_msg)
                st.session_state.last_error = error_msg
                return False
        
        # Initialize the QA system
        with st.spinner("🧠 Setting up AI system..."):
            if setup_qa_system():
                st.session_state.papers_loaded = True
                st.success(f"🎉 Successfully loaded {processed_count} papers with {len(st.session_state.docs)} text chunks")
                return True
            else:
                error_msg = "❌ Failed to initialize QA system"
                st.error(error_msg)
                return False
                
    except Exception as e:
        error_msg = f"Error in fetch_papers: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.session_state.last_error = error_msg
        
        # Show detailed error in expander
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc())
        
        return False

def setup_qa_system():
    """Initialize the QA system with loaded documents and better error handling"""
    if not st.session_state.docs:
        error_msg = "No documents loaded. Please load papers first."
        st.error(f"❌ {error_msg}")
        st.session_state.last_error = error_msg
        return False
        
    try:
        # Get embedder
        embedder = get_embedder()
        if embedder is None:
            raise Exception("Failed to initialize embedder")
        
        # Build vectorstore
        vectorstore = build_vectorstore(st.session_state.docs, embedder)
        if vectorstore is None:
            raise Exception("Failed to build vectorstore")
        
        # Test vectorstore
        test_results = vectorstore.similarity_search("test", k=1)
        if not test_results:
            raise Exception("Vectorstore appears to be empty")
        
        # Create custom prompt
        prompt_template = """You are a helpful AI research assistant analyzing research papers. 

Context from research papers:
{context}

Chat History: {chat_history}

Human Question: {question}

Instructions:
- Provide accurate, detailed answers based ONLY on the provided context
- If information isn't available in the context, clearly state "Based on the provided papers, I don't have information about..."
- Cite specific papers when possible
- Keep responses informative but under 300 words
- Be conversational and helpful

Answer:"""

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        # Test LLM
        test_response = llm.invoke("Hello")
        if not test_response:
            raise Exception("LLM test failed")
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            return_source_documents=True,
            verbose=True
        )
        
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain
        
        return True
        
    except Exception as e:
        error_msg = f"Error setting up QA system: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.session_state.last_error = error_msg
        st.session_state.qa_chain = None
        st.session_state.vectorstore = None
        
        # Show detailed error
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc())
        
        return False

def display_loaded_papers():
    """Display the loaded research papers"""
    if not st.session_state.titles:
        return
    
    st.markdown("## 📚 Loaded Research Papers")
    
    for i, title in enumerate(st.session_state.titles):
        st.markdown(f"""
        <div class="paper-container">
            <strong>Paper {i+1}:</strong> {title}
        </div>
        """, unsafe_allow_html=True)

def render_conversation_interface():
    """Render the main conversation interface"""
    if not st.session_state.papers_loaded:
        st.info("📋 Please load some research papers first to start the conversation")
        return "", False
    
    st.markdown("## 💬 Research Conversation")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Agent:</strong> {answer}
            </div>
            """, unsafe_allow_html=True)
    
    # Question input
    st.markdown("### ❓ Ask Your Question")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        question = st.text_area(
            "Your Question:",
            placeholder="What would you like to know about these research papers?",
            height=100,
            key="question_input"
        )
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        ask_button = st.button("🧠 Ask", type="primary", use_container_width=True)
    
    return question, ask_button

def map_paper_references(question):
    """Replace 'paper 1', 'paper 2', etc. with actual paper titles in the question"""
    def replacer(match):
        paper_num = int(match.group(1))
        if 0 < paper_num <= len(st.session_state.titles):
            return st.session_state.titles[paper_num - 1]  # update sr-2
        return match.group(0)  # fallback
    
    return re.sub(r'paper\s*(\d+)', replacer, question, flags=re.IGNORECASE)  



def process_question(question):
    """Process user question and generate response with better error handling"""
    if not question.strip():
        st.warning("⚠️ Please enter a question")
        return
    
    # Clear previous error
    st.session_state.last_error = None
    
    # Ensure QA system is initialized
    if st.session_state.qa_chain is None:
        st.warning("🔄 QA system not ready. Attempting to initialize...")
        if not setup_qa_system():
            return
    
    try:
        with st.spinner("🤔 Analyzing papers and generating response..."):
            # Process the question

            mapped_question = map_paper_references(question)
            result = st.session_state.qa_chain.invoke({"question": mapped_question})
            
            if not result:
                raise Exception("No result returned from QA chain")
            
            answer = result.get("answer", "No answer generated")
            sources = result.get("source_documents", [])
            
            if not answer or answer.strip() == "":
                raise Exception("Empty answer generated")
            
            # Display the answer
            st.markdown("### 🎯 Response")
            st.markdown(f"""
            <div class="chat-message assistant-message">
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if sources:
                st.markdown("### 📚 Referenced Sources")
                unique_sources = []
                for src in sources:
                    source_name = src.metadata.get('source', 'Unknown')
                    if source_name not in unique_sources:
                        unique_sources.append(source_name)
                
                for source in unique_sources:
                    st.markdown(f"""
                    <div class="source-item">
                        📄 {source}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Save to chat history
            st.session_state.chat_history.append((mapped_question, answer))
            st.session_state.conversation_active = True
            
            # Show continuation prompt
            st.markdown("""
            <div class="continue-prompt">
                💭 Feel free to ask follow-up questions or explore different aspects of these research papers!
            </div>
            """, unsafe_allow_html=True)
            
            # Clear the input
            st.rerun()
            
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.session_state.last_error = error_msg
        
        # Show detailed error
        with st.expander("🔍 Detailed Error Information"):
            st.code(traceback.format_exc())
        
        # Try to reinitialize the QA system
        st.info("🔄 Attempting to reinitialize QA system...")
        setup_qa_system()

def render_conversation_controls():
    """Render conversation control buttons"""
    if st.session_state.conversation_active:
        st.markdown("### 🛠️ Conversation Controls")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 New Topic", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_active = False
                st.rerun()
        
        with col2:
            if st.button("📋 View Summary", use_container_width=True):
                show_conversation_summary()
        
        with col3:
            if st.button("💾 Save Chat", use_container_width=True):
                save_conversation()

def show_conversation_summary():
    """Display a summary of the conversation"""
    if st.session_state.chat_history:
        with st.expander("📋 Conversation Summary", expanded=True):
            st.markdown(f"**Total Questions Asked:** {len(st.session_state.chat_history)}")
            st.markdown(f"**Papers Referenced:** {len(st.session_state.titles)}")
            
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a[:100]}..." if len(a) > 100 else f"**A{i+1}:** {a}")
                st.markdown("---")

def save_conversation():
    """Save conversation to a downloadable format"""
    if st.session_state.chat_history:
        conversation_text = "# Research Paper Conversation\n\n"
        conversation_text += f"**Papers Analyzed:** {len(st.session_state.titles)}\n\n"
        
        for title in st.session_state.titles:
            conversation_text += f"- {title}\n"
        
        conversation_text += "\n## Conversation History\n\n"
        
        for i, (q, a) in enumerate(st.session_state.chat_history):
            conversation_text += f"**Question {i+1}:** {q}\n\n"
            conversation_text += f"**Answer {i+1}:** {a}\n\n"
            conversation_text += "---\n\n"
        
        st.download_button(
            label="💾 Download Conversation",
            data=conversation_text,
            file_name="research_conversation.md",
            mime="text/markdown"
        )





def main():
    """Main application function"""
    configure_app()
    initialize_session()
    
    # Check environment first
    if not check_environment():
        st.stop()
    
    render_sidebar()
    
    # Main content
    render_header()
    
    # Paper search section
    query, search_button = render_search_section()
    
    if search_button:
        fetch_papers(query)
    
    # Display loaded papers
    display_loaded_papers()
    
    st.markdown("---")
    
    # Conversation interface
    question, ask_button = render_conversation_interface()
    
    if ask_button and question:
        process_question(question)
    
    # Conversation controls
    render_conversation_controls()

if __name__ == "__main__":
    main()