import streamlit as st
import requests
import base64
import os
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GitHub Repo Reader",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🚀 GitHub Repository Analyzer")
st.markdown("*Leverage AI to understand GitHub repositories at a glance*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check API keys
    groq_key = os.getenv("GROQ_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Groq API", "✅ Loaded" if groq_key else "❌ Missing")
    with col2:
        st.metric("GitHub Token", "✅ Loaded" if github_token else "❌ Missing")
    
    st.divider()
    
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.number_input("Max Tokens", 256, 4096, 2048, 256)
    max_files = st.number_input("Max Files to Analyze", 5, 50, 20)
    context_size = st.number_input("Context Retrieved (k)", 1, 10, 5)

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository",
        key="repo_input"
    )

with col2:
    analyze_button = st.button("📊 Analyze", use_container_width=True)

# Helper functions
GITHUB_API = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {github_token}"
} if github_token else {}

def parse_repo_url(url):
    """Extract owner and repo name from GitHub URL"""
    try:
        parts = url.rstrip("/").split("/")
        return parts[-2], parts[-1]
    except:
        return None, None

def get_readme(owner, repo):
    """Fetch README content from repository"""
    try:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/readme"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""
        content = r.json()["content"]
        return base64.b64decode(content).decode("utf-8")
    except Exception as e:
        st.warning(f"Could not fetch README: {str(e)}")
        return ""

def get_repo_files(owner, repo, branch="main"):
    """Fetch all files in repository"""
    try:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        return [f["path"] for f in r.json()["tree"] if f["type"] == "blob"]
    except Exception as e:
        st.warning(f"Could not fetch files: {str(e)}")
        return []

def get_file_content(owner, repo, path):
    """Fetch content of a specific file"""
    try:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""
        content = r.json().get("content", "")
        return base64.b64decode(content).decode("utf-8", errors="ignore")
    except:
        return ""

def load_repo_documents(owner, repo, files, max_files=20):
    """Load and prepare repository documents"""
    IMPORTANT_EXT = (".py", ".js", ".ts", ".md", ".ipynb", ".json", ".yaml", ".yml")
    docs = []
    progress_bar = st.progress(0)
    
    for idx, file in enumerate(files):
        if file.endswith(IMPORTANT_EXT):
            text = get_file_content(owner, repo, file)
            if text.strip():
                docs.append({
                    "id": file,
                    "text": text[:4000]  # limit per file
                })
            
            progress = min((idx + 1) / len(files), 1.0)
            progress_bar.progress(progress)
        
        if len(docs) >= max_files:
            break
    
    progress_bar.empty()
    return docs

def embed_and_store(docs, collection):
    """Embed documents and store in ChromaDB"""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    progress_bar = st.progress(0)
    
    for idx, doc in enumerate(docs):
        embedding = embedding_model.encode(doc["text"]).tolist()
        collection.add(
            documents=[doc["text"]],
            embeddings=[embedding],
            ids=[doc["id"]]
        )
        progress_bar.progress((idx + 1) / len(docs))
    
    progress_bar.empty()

def retrieve_context(collection, query, k=5):
    """Retrieve relevant context from collection"""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    return "\n\n".join(results["documents"][0]) if results["documents"] else ""

def generate_repo_summary(llm, prompt, context: str) -> str:
    """Generate repository summary using LLM"""
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context})
    return result

# Create prompt template
repo_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are a senior software engineer and technical documentation expert.

Using the repository context below, generate a comprehensive analysis including:
1. **Project Overview**: What is this project about?
2. **Tech Stack**: What technologies and frameworks are used?
3. **Core Features**: What are the main features and functionalities?
4. **Project Structure**: How is the code organized?
5. **How to Run**: Setup and execution instructions
6. **Key Dependencies**: Important libraries and their purposes
7. **Use Cases**: What problems does this project solve?

Repository Context:
{context}

Provide a clear, well-structured analysis that would help a new developer quickly understand the project.
"""
)

# Analysis logic
if analyze_button and repo_url:
    if not groq_key or not github_token:
        st.error("❌ Missing API credentials. Please set GROQ_API_KEY and GITHUB_TOKEN in your .env file")
    else:
        with st.spinner("🔍 Analyzing repository..."):
            # Parse repository URL
            owner, repo = parse_repo_url(repo_url)
            
            if not owner or not repo:
                st.error("❌ Invalid GitHub URL format. Please use: https://github.com/owner/repo")
            else:
                try:
                    # Initialize LLM
                    llm = ChatGroq(
                        model_name="llama-3.3-70b-versatile",
                        groq_api_key=groq_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=None,
                        max_retries=2
                    )
                    
                    # Load repository content
                    st.info(f"📦 Loading repository: {owner}/{repo}")
                    readme = get_readme(owner, repo)
                    st.success("✅ README fetched")
                    
                    files = get_repo_files(owner, repo)
                    st.info(f"📄 Found {len(files)} files in repository")
                    
                    docs = load_repo_documents(owner, repo, files, max_files)
                    st.success(f"✅ Loaded {len(docs)} important files")
                    
                    if readme:
                        docs.insert(0, {"id": "README.md", "text": readme[:4000]})
                    
                    # Reset and store embeddings
                    st.info("🧠 Creating embeddings...")
                    chroma_client = chromadb.Client()
                    collection = chroma_client.create_collection(name="repo_docs")
                    embed_and_store(docs, collection)
                    st.success("✅ Embeddings created and stored")
                    
                    # Retrieve context
                    st.info("🔎 Retrieving relevant context...")
                    query = "Explain this GitHub repository comprehensively"
                    context = retrieve_context(collection, query, k=context_size)
                    st.success("✅ Context retrieved")
                    
                    # Generate explanation
                    st.info("✨ Generating AI analysis...")
                    analysis_result = generate_repo_summary(llm, repo_prompt, context)
                    st.session_state.analysis_result = analysis_result
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

# Display results
if st.session_state.analysis_result:
    st.divider()
    
    st.subheader("📋 Repository Analysis Results")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Full Analysis", "📝 Raw Text", "📥 Download"])
    
    with tab1:
        st.markdown(st.session_state.analysis_result)
    
    with tab2:
        st.text_area("Raw Analysis Text", value=st.session_state.analysis_result, height=400)
    
    with tab3:
        # Create downloadable content
        analysis_text = st.session_state.analysis_result
        repo_name = repo_url.split("/")[-1] if repo_url else "analysis"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download as Text",
                data=analysis_text,
                file_name=f"{repo_name}_analysis.txt",
                mime="text/plain"
            )
        
        with col2:
            # Convert to markdown for better formatting
            markdown_content = f"# Repository Analysis: {repo_url}\n\n{analysis_text}"
            st.download_button(
                label="📑 Download as Markdown",
                data=markdown_content,
                file_name=f"{repo_name}_analysis.md",
                mime="text/markdown"
            )

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>🚀 GitHub Repository Analyzer powered by Groq API and LangChain</p>
    </div>
    """, unsafe_allow_html=True)
