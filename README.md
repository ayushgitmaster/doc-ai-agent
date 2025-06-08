# 🤖 Doc Q&A AI Research Agent

An interactive research companion powered by LangChain + Gemini + arXiv, allowing you to search, load, and chat with academic papers like a pro.  
Built using Python and Streamlit.

---

## 🚀 Features

- 🔍 Search research papers from **arXiv**
- 📚 Auto-fetch and parse PDF content
- 🧠 Intelligent Q&A with **LangChain + Gemini**
- 💬 Chat-based interface to ask questions about papers
- 🎯 Support for queries like:
  - `"Summarize paper 1"`
  - `"What methodology is used in paper 2?"`
- 📄 Save and export your conversation as markdown
- 🔁 Continue conversations contextually

---

## 📦 Tech Stack

| Tool           | Purpose                          |
|----------------|----------------------------------|
| **Streamlit**  | UI for web app                   |
| **LangChain**  | Manages memory, retrieval, and chaining |
| **Gemini API** | LLM from Google (via `ChatGoogleGenerativeAI`) |
| **arXiv API**  | For fetching relevant research papers |
| **PDF Parsing**| Extracts text from downloaded papers |
| **Vector DB**  | In-memory document chunk storage |

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
 

### 2. . Install dependencies

pip install -r requirements.txt


### 3. Set environment variables

Create a .env file:
GOOGLE_API_KEY=your_google_api_key_here

### 4. Run the App

streamlit run main.py