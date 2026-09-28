import os
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest
import streamlit as st
import rag_runtime

QUESTION = 'What procedures are described for knee arthroscopy?'
DOCS = [Document(page_content=f'Sample excerpt {i}', metadata={'row_id': i, 'title': f'Record {i}', 'specialty': 'Surgery'}) for i in range(4)]

class Store:
    def __init__(self): self.calls = []
    def similarity_search_with_score(self, question, k):
        self.calls.append((question, k))
        return [(d, float(i)/10) for i, d in enumerate(DOCS)]

class PortfolioFlowTests(unittest.TestCase):
    def setUp(self): st.cache_resource.clear()
    def tearDown(self): st.cache_resource.clear()

    def test_validation_and_retrieval_metadata(self):
        store = Store()
        for question in ['', ' ', 'x'*1001]:
            with self.assertRaises(ValueError): rag_runtime.retrieve(store, question)
        result = rag_runtime.retrieve(store, QUESTION)
        self.assertEqual(len(result['documents']), 4)
        self.assertEqual(result['question'], QUESTION)
        self.assertEqual(store.calls, [(QUESTION, 4)])

    def test_provider_errors_preserve_evidence_and_retry(self):
        from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted, DeadlineExceeded, PermissionDenied
        for error in [ServiceUnavailable('SECRET_TOKEN'), ResourceExhausted('SECRET_TOKEN'), DeadlineExceeded('SECRET_TOKEN'), PermissionDenied('SECRET_TOKEN'), TimeoutError('SECRET_TOKEN')]:
            with self.subTest(error=type(error).__name__):
                st.cache_resource.clear()
                store, calls = Store(), []
                class Service:
                    def __init__(self, *args): pass
                    def generate(self, evidence):
                        calls.append(evidence)
                        if len(calls) == 1: raise error
                        return 'Source-supported test answer [record 0]'
                with patch.dict(os.environ, {'GOOGLE_API_KEY':'TEST_KEY_DO_NOT_DISPLAY','GEMINI_MODEL':'test-model'}), patch.object(rag_runtime,'load_vectorstore',return_value=store), patch.object(rag_runtime,'AnswerService',Service):
                    app=AppTest.from_file('app.py',default_timeout=30).run()
                    self.assertFalse(app.exception)
                    app.button[1].click().run()
                    self.assertIn('Enter a question first.', [w.value for w in app.warning])
                    app.button[0].click().run(); app.button[1].click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(len(app.session_state['evidence']['documents']),4)
                    self.assertEqual(len(app.expander),5)
                    visible=' '.join(str(e.value) for e in app.warning)+str(app)
                    self.assertNotIn('SECRET_TOKEN',visible)
                    self.assertNotIn('TEST_KEY_DO_NOT_DISPLAY',visible)
                    self.assertEqual(len(calls),1)
                    self.assertTrue(any(b.label=='Retry answer' for b in app.button))
                    app.text_area[0].set_value('A different unsubmitted question')
                    next(b for b in app.button if b.label=='Retry answer').click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state['answer'],'Source-supported test answer [record 0]')
                    self.assertEqual(len(store.calls),1)
                    self.assertEqual(calls[1]['question'],QUESTION)
                    self.assertEqual(len(app.expander),5)
                    self.assertFalse(app.warning)

    def test_missing_key_still_retrieves(self):
        with patch.dict(os.environ, {'GOOGLE_API_KEY':'','GEMINI_MODEL':''}), patch.object(rag_runtime,'load_vectorstore',return_value=Store()):
            app=AppTest.from_file('app.py').run()
            app.text_area[0].set_value('How do I repair a bicycle tire?')
            app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state['evidence']['documents']),4)
            self.assertTrue(app.warning)

    def test_runtime_generation_uses_existing_documents_and_bounds_retries(self):
        from langchain_core.language_models.fake import FakeListLLM
        store=Store()
        # The original chain's retriever requires a BaseRetriever; use a real in-memory stub.
        from langchain_core.retrievers import BaseRetriever
        class Retriever(BaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager):
                raise AssertionError('Generation must not retrieve again')
        store.as_retriever=lambda **kwargs: Retriever()
        with patch('langchain_google_genai.ChatGoogleGenerativeAI',return_value=FakeListLLM(responses=['Test answer [record 0]'])):
            service=rag_runtime.AnswerService(store,'test','test')
            evidence=rag_runtime.retrieve(store,QUESTION)
            self.assertEqual(service.generate(evidence),'Test answer [record 0]')
            with self.assertRaisesRegex(RuntimeError,'COOLDOWN'): service.generate(evidence)
            self.assertEqual(len(store.calls),1)

if __name__=='__main__': unittest.main()
