"""Educational RAG demo: preserve retrieval evidence independently of generation."""
import logging
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from rag_runtime import AnswerService, load_vectorstore, retrieve, MAX_QUESTION_LENGTH

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
# Provider retry logs may contain raw API error text. Log only exception types here.
logging.getLogger('langchain_google_genai').setLevel(logging.CRITICAL)
st.set_page_config(page_title='Medical QA RAG', page_icon='📚', layout='wide')
st.title('Medical QA RAG')
st.write('Explore historical medical transcriptions with source-referenced question answering.')
st.caption('Educational demonstration only. This system retrieves from a historical medical-transcription dataset and is not a substitute for professional medical advice.')
st.caption('An individual university project by Faizan Tariq.')
st.info('Do not enter personal health information. Questions and retrieved excerpts are sent to Google Gemini; free-tier content may be used to improve Google products.')


def setting(name):
    if name in os.environ:
        return os.environ[name]
    if st.secrets.load_if_toml_exists():
        return str(st.secrets.get(name, ''))
    return ''


@st.cache_resource(show_spinner=False)
def cached_store():
    return load_vectorstore()


@st.cache_resource(show_spinner=False)
def cached_service(api_key, model):
    return AnswerService(cached_store(), api_key, model)


def use_example():
    st.session_state['question'] = 'What procedures are described for knee arthroscopy?'


def generate_answer():
    evidence = st.session_state['evidence']
    st.session_state.pop('answer', None)
    api_key, model = setting('GOOGLE_API_KEY'), setting('GEMINI_MODEL')
    if not api_key or not model:
        st.session_state['generation_notice'] = 'Answer generation is not configured. You can still explore the retrieved sources below.'
        return
    try:
        with st.spinner('Generating an answer from the retrieved sources…'):
            st.session_state['answer'] = cached_service(api_key, model).generate(evidence)
        st.session_state.pop('generation_notice', None)
    except Exception as exc:
        logging.error('Medical RAG generation failed (%s)', type(exc).__name__)
        if isinstance(exc, RuntimeError) and str(exc) in ('BUSY', 'COOLDOWN'):
            notice = 'The shared demo is busy. Your retrieved sources are preserved; please retry shortly.'
        else:
            notice = 'The generation service is temporarily unavailable. The relevant retrieved sources are shown below; please try generating the answer again shortly.'
        st.session_state['generation_notice'] = notice


st.button('Use example question', on_click=use_example)
with st.form('question_form'):
    question = st.text_area('Question about the sample documents', key='question', height=110,
                            max_chars=MAX_QUESTION_LENGTH,
                            placeholder='What procedures are described for knee arthroscopy?')
    submitted = st.form_submit_button('Get answer', type='primary')

if submitted:
    if not question.strip():
        st.warning('Enter a question first.')
    else:
        for name in ('evidence', 'answer', 'generation_notice'):
            st.session_state.pop(name, None)
        try:
            with st.spinner('Retrieving relevant excerpts…'):
                st.session_state['evidence'] = retrieve(cached_store(), question)
        except Exception as exc:
            logging.error('Medical RAG retrieval failed (%s)', type(exc).__name__)
            st.error('The source collection could not be loaded. Please try again later.')
        else:
            generate_answer()

if 'evidence' in st.session_state:
    evidence = st.session_state['evidence']
    st.subheader('Results for your submitted question')
    st.write(evidence['question'])
    if 'answer' not in st.session_state:
        if st.button('Retry answer', help='Reuse the same question and retrieved excerpts.'):
            generate_answer()
    if 'generation_notice' in st.session_state:
        st.warning(st.session_state['generation_notice'])
    if 'answer' in st.session_state:
        st.subheader('Answer')
        st.write(st.session_state['answer'])
        st.caption('Generated text can contain errors. Compare each claim with the retrieved excerpts below.')
        st.download_button('Download answer', st.session_state['answer'], 'medical-rag-answer.txt', mime='text/plain')
    st.subheader('Retrieved sources')
    st.caption(f"{len(evidence['documents'])} chunks · MiniLM embeddings · FAISS similarity retrieval")
    for i, (doc, distance) in enumerate(zip(evidence['documents'], evidence['distances']), 1):
        title = str(doc.metadata.get('title', 'Untitled')).strip()
        record = doc.metadata.get('row_id', 'Unknown')
        with st.expander(f'{i}. {title} — record {record}'):
            st.text(f"Specialty: {doc.metadata.get('specialty', 'Unknown')}")
            st.caption(f'Retrieval rank: {i} · Squared L2 distance: {distance:.3f} (lower is nearer; not a confidence score)')
            st.text(doc.page_content)
    st.caption('Nearest neighbors can be irrelevant, and multiple chunks may repeat the same record. Source references do not establish clinical correctness.')

with st.expander('About this project'):
    st.write('Question → MiniLM embedding → FAISS retrieval → top-k clinical transcription chunks → Gemini → answer with source references')
    st.write('4,966 source records; 29,598 saved chunks; 384-dimensional vectors. The recovered dataset and index are unchanged.')
    st.caption('No clinical-performance benchmark has been established. The application does not write questions or answers to disk. Retrieval remains visible when the generation provider is unavailable.')
