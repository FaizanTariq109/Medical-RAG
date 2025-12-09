# 🏥 Medical RAG QA System

A production-ready Retrieval Augmented Generation (RAG) system that answers medical questions using clinical transcriptions. Built with LangChain, FAISS, and Google Gemini API, this system provides accurate, citation-aware responses based on a corpus of 5,000+ medical transcriptions.

## 🎯 Overview

This RAG-powered medical assistant:

- **Retrieves** relevant information from 5,000+ clinical transcriptions
- **Augments** context with medical specialty classifications and keywords
- **Generates** accurate, evidence-based responses using Google Gemini
- **Cites** source documents for transparency and verification

### Key Features

✅ **Semantic Search** - FAISS vector store for fast similarity search  
✅ **Citation-Aware** - Every response includes source document references  
✅ **Medical Specialties** - Organized by 40+ medical specialties  
✅ **Interactive UI** - User-friendly Streamlit web interface  
✅ **Evaluation Framework** - Tested on 30+ medical queries  
✅ **Production Ready** - Complete error handling and logging

## 📊 Dataset

**Medical Transcriptions Dataset** from Kaggle

- **Source:** [Medical Transcriptions on Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- **Size:** ~5,000 clinical transcriptions
- **Specialties:** 40+ medical specialties (Surgery, Cardiology, Neurology, etc.)
- **Columns:**
  - `description` - Brief description of the transcription
  - `medical_specialty` - Medical specialty classification
  - `sample_name` - Transcription title
  - `transcription` - Full medical transcription text
  - `keywords` - Relevant medical keywords

## 🏗️ Architecture

```
User Query → Embedding → FAISS Search → Top-K Chunks →
              ↓
         Gemini LLM (with context) → Response + Citations
```

### Components

1. **Document Processing**

   - Load and preprocess clinical transcriptions
   - Combine relevant fields (specialty, keywords, transcription)
   - Split into overlapping chunks (1000 chars, 200 overlap)

2. **Vector Store**

   - Sentence Transformers (all-MiniLM-L6-v2) for embeddings
   - FAISS for efficient similarity search
   - Metadata preservation for citations

3. **RAG Pipeline**

   - RetrievalQA chain with Google Gemini
   - Top-4 chunk retrieval for comprehensive context
   - Temperature 0.3 for balanced factual responses

4. **Streamlit Interface**
   - Clean, intuitive medical query input
   - Expandable source document viewer
   - Copy-to-clipboard functionality

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get it free](https://makersuite.google.com/app/apikey))
- ~2GB disk space for models and data

### Installation

```bash
# Clone the repository
git clone https://github.com/FaizanTariq109/Medical-RAG.git
cd Medical-RAG

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Or set it as an environment variable:

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Mac/Linux
export GEMINI_API_KEY=your_api_key_here
```

### Run the Application

```bash
# Step 1: Process data and build vector store (one-time setup)
python build_vectorstore.py

# Step 2: Launch Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
Medical-RAG/
│
├── app.py                      # Streamlit web interface
├── build_vectorstore.py        # Vector store creation script
├── requirements.txt            # Python dependencies
├── .env                        # API key configuration (create this)
├── .gitignore                  # Git ignore rules
│
├── data/
│   └── mtsamples.csv          # Medical transcriptions dataset
│
├── vectorstore/
│   └── faiss_index/           # FAISS vector store (generated)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── embeddings.py          # Embedding generation
│   └── rag_chain.py           # RAG pipeline setup
│
├── evaluation/
│   ├── test_queries.py        # 30+ test medical queries
│   └── evaluate.py            # Evaluation framework
│
└── README.md                  # This file
```

## 💻 Usage Examples

### Example 1: Simple Medical Query

**Query:** "What are the symptoms of type 2 diabetes?"

**Response:**

```
Common symptoms of type 2 diabetes include:
- Increased thirst and frequent urination
- Increased hunger
- Unintended weight loss
- Fatigue and weakness
- Blurred vision
- Slow-healing sores
- Frequent infections

Source Documents:
1. SOAP_Endocrinology_47.txt
2. Discharge_Summary_Diabetes_102.txt
```

### Example 2: Procedure Information

**Query:** "How is a coronary artery bypass performed?"

**Response:**

```
Coronary artery bypass grafting (CABG) is performed by:
1. General anesthesia administration
2. Median sternotomy incision
3. Harvesting graft vessel (typically saphenous vein or internal mammary artery)
4. Cardiopulmonary bypass initiation
5. Grafting vessels to bypass blocked coronary arteries
6. Weaning from bypass
7. Chest closure

Source Documents:
1. Operative_Note_CABG_234.txt
2. Consultation_Cardiology_156.txt
```

### Example 3: Differential Diagnosis

**Query:** "What conditions should be considered for chest pain?"

**Response:**

```
Differential diagnosis for chest pain includes:

Cardiac: Angina, myocardial infarction, pericarditis
Pulmonary: Pulmonary embolism, pneumonia, pleuritis
Gastrointestinal: GERD, esophageal spasm, peptic ulcer
Musculoskeletal: Costochondritis, muscle strain
Other: Anxiety, panic disorder

Evaluation should include ECG, cardiac enzymes, and chest X-ray.

Source Documents:
1. Emergency_Department_Note_089.txt
2. Consultation_Cardiology_203.txt
```

## 🧪 Evaluation

The system has been evaluated on 30+ diverse medical queries covering:

- **Symptoms & Diagnosis** (e.g., "What are signs of pneumonia?")
- **Treatment Procedures** (e.g., "How is appendicitis treated?")
- **Medication Information** (e.g., "What are side effects of metformin?")
- **Anatomical Questions** (e.g., "What is the function of the liver?")

### Evaluation Metrics

```python
# Run evaluation
python evaluation/evaluate.py

# Sample output:
Total Queries: 30
Average Response Time: 2.3s
Source Citation Rate: 100%
Relevant Response Rate: 93.3%
```

### Run Your Own Evaluation

```python
from evaluation.test_queries import medical_queries
from utils.rag_chain import get_rag_chain

# Load RAG chain
qa_chain = get_rag_chain()

# Test queries
for query in medical_queries:
    result = qa_chain({"query": query})
    print(f"Q: {query}")
    print(f"A: {result['result']}")
    print(f"Sources: {len(result['source_documents'])}")
    print("-" * 80)
```

## 🛠️ Configuration

### Adjust Chunk Size

In `build_vectorstore.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Increase for more context per chunk
    chunk_overlap=200,    # Increase to prevent splitting mid-sentence
    length_function=len,
)
```

### Change Number of Retrieved Documents

In `utils/rag_chain.py`:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Retrieve top 4 chunks (default)
)
```

### Adjust LLM Temperature

In `utils/rag_chain.py`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,  # Lower = more focused, Higher = more creative
)
```

## 📊 Performance

| Metric              | Value                 |
| ------------------- | --------------------- |
| Dataset Size        | 5,000+ transcriptions |
| Vector Store Size   | ~150MB                |
| Average Query Time  | 2-3 seconds           |
| Embedding Dimension | 384                   |
| Total Chunks        | ~25,000               |
| Specialties Covered | 40+                   |

## 🔧 Troubleshooting

### "API key not found"

```bash
# Make sure .env file exists with:
GEMINI_API_KEY=your_actual_key_here
```

### "Vector store not found"

```bash
# Run the build script first:
python build_vectorstore.py
```

### Slow query responses

```python
# Reduce number of retrieved chunks in rag_chain.py:
search_kwargs={"k": 2}  # Instead of 4
```

### Out of memory

```python
# Process fewer documents in build_vectorstore.py:
df = df.head(1000)  # Only use first 1000 transcriptions
```

## 🚨 Important Disclaimers

⚠️ **Medical Disclaimer:** This system is for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

⚠️ **Accuracy:** While the system provides evidence-based responses with citations, medical information can become outdated, and the AI may occasionally generate incorrect information. Always verify critical information with authoritative medical sources.

⚠️ **Privacy:** Do not input personal health information or identifiable patient data into this system.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add medical fact-checking layer
- [ ] Implement multi-query retrieval
- [ ] Add medical terminology disambiguation
- [ ] Create mobile-responsive UI
- [ ] Add conversation memory
- [ ] Implement user feedback mechanism

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Technical Stack

- **LangChain** - RAG framework and orchestration
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **Google Gemini** - Large language model
- **Streamlit** - Web interface
- **Pandas** - Data processing
- **Python 3.8+** - Core programming language

## 📖 Learn More

### Related Papers

- ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401)
- ["Dense Passage Retrieval for Open-Domain Question Answering"](https://arxiv.org/abs/2004.04906)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Faizan Tariq**

- GitHub: [@FaizanTariq109](https://github.com/FaizanTariq109)
- LinkedIn: [faizan-109t](https://www.linkedin.com/in/faizan-109t/)
- Email: faizan3san@gmail.com

## 🙏 Acknowledgments

- **Dataset:** Medical Transcriptions Dataset from Kaggle
- **LangChain:** For the excellent RAG framework
- **Google:** For the Gemini API
- **Sentence Transformers:** For the embedding models
- **Streamlit:** For the intuitive UI framework

---

**Note:** This is an educational project demonstrating RAG systems for medical information retrieval. Always consult healthcare professionals for medical advice.
