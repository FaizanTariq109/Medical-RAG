import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical QA Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Medical QA Assistant")
st.markdown("""
This RAG-powered system answers medical questions based on clinical transcriptions.
Ask questions about symptoms, procedures, diagnoses, or treatments.
""")

# Load API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
    st.stop()

# Initialize components (cached to avoid reloading)
@st.cache_resource
def load_rag_system():
    """Load the vector store and create RAG chain"""
    
    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the saved vector store
    vectorstore = FAISS.load_local(
        "faiss_medical_index",
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading pickled data
    )
    
    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

# Load the system
try:
    with st.spinner("Loading RAG system..."):
        qa_chain = load_rag_system()
    st.success("✓ System ready!")
except Exception as e:
    st.error(f"Error loading system: {str(e)}")
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # User input
    question = st.text_area(
        "Enter your medical question:",
        height=100,
        placeholder="e.g., What are the symptoms of type 2 diabetes?"
    )
    
    # Ask button
    ask_button = st.button("🔍 Get Answer", type="primary")

with col2:
    st.markdown("### Example Questions:")
    st.markdown("""
    - What are symptoms of diabetes?
    - How is hypertension treated?
    - What procedures are used for knee surgery?
    - What are complications of pneumonia?
    """)

# Process question
if ask_button and question:
    with st.spinner("Searching medical records and generating answer..."):
        try:
            result = qa_chain({"query": question})
            
            # Display answer
            st.markdown("### 💡 Answer:")
            st.info(result['result'])
            
            # Display sources
            st.markdown("### 📚 Source Documents:")
            for i, doc in enumerate(result['source_documents'], 1):
                with st.expander(f"Source {i}: {doc.metadata.get('title', 'Unknown')}"):
                    st.markdown(f"**Specialty:** {doc.metadata.get('specialty', 'Unknown')}")
                    st.markdown(f"**Keywords:** {doc.metadata.get('keywords', 'N/A')}")
                    st.markdown("**Content:**")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")

elif ask_button:
    st.warning("⚠️ Please enter a question first!")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain, FAISS, and Google Gemini*")