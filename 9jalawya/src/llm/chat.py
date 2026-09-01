# app/rag.py
import time
from typing import Optional 
from langchain_core.output_parsers import StrOutputParser
from src.llm.llm_engine import model
from src.llm.prompt import sys_prompt
from src.utils.logging_config import get_logger, setup_logging
from src.retrieval.retriever import NaijaLawRetriever, RetrievalFilter
from src.config.settings import Settings
from src.agent.intent_detector import detect_intent, Intent, IntentResult
from src.agent.web_search import get_search_tool

setup_logging()

logger = get_logger(__name__)

settings = Settings()
class Lawya:

    def __init__(self):
        logger.info("Initialising 9jaLawya...")

        logger.info("Loading retriever and vector store...")
        self.retriever = NaijaLawRetriever()

        model_start = time.time()
        try:
            logger.info("Loading LLM...")
            self.llm = model()
        except Exception as e:
            logger.exception(f"Failed to load LLM: {e}")
            raise

        model_elapsed = round(time.time() - model_start, 1)
        logger.info(f"Loaded model in {model_elapsed}s")
        self.get_rag_chain = self._get_rag_chain()

    def _format_context(self, results: list) -> str:
        """
        Format the retrieved results into a context string.
        Args:
            results: The retrieved results to format.
        Returns:
            str: The formatted context string.
        """
        return "\n\n---\n\n".join(
            r.to_context_string() for r in results
        )

    def _get_rag_chain(self):
        """
        Build and cache the RAG chain. Called once at startup.

        Returns:
         Returns a runnable chain that takes a question and returns an answer.
         """
        logger.info("Initialising RAG chain...")

        # prompt 
        prompt_template = sys_prompt()

        rag_chain = (prompt_template| self.llm| StrOutputParser())

        logger.info("RAG chain ready.")
        return rag_chain

    def answer_query(self, query: str) -> dict:

        """
        Retrieve context and generate an answer. Returns a dict with
        answer text and a flag indicating whether it succeeded.
    
        Args:
          query (str): The query to be answered.
    
        Returns:
        dict: A dictionary containing the answer, success flag, and sources.
        """
        if not query or not query.strip():
            return{
                "answer": "Please provide a prompt",
                "Success": False, 
            }

        try:
            logger.info(f'Incoming query: {query[:80]}')
        
            # STEP 1: DETECT INTENT
            chain = self.get_rag_chain
            logger.info("RAG chain invoked")

            intent = detect_intent(query)
            logger.info(
                f"Intent: {intent.intent.value} "
                f"(confidence: {intent.confidence:.2f}, "
                f"domain: {intent.legal_domain})"
            )

            should_search = intent.requires_websearch
            web_context = ""

            if should_search:
                logger.info(f"Intent requires web search: [{intent.reason_websearch}]. Performing search...")
                search_results = get_search_tool().search(query)
                if search_results.success and search_results.results:
                    web_context = search_results.to_context_string()
                    logger.info(f"Web search completed. Web search returned {len(search_results.results)} results.")

            # STEP 2: RETRIEVE LEGAL CONTEXT
            if intent.intent != Intent.GENERAL_CHAT:
                filters = self._build_filters(intent)

                retrieval_query = intent.search_query or query

                logger.info(f"Retrieving legal context using query: {retrieval_query[:100]}")
                
                results = self.retriever.retrieve(
                                                query=retrieval_query,
                                                top_k=settings.retriever.top_k, 
                                                similarity_threshold=settings.retriever.similarity_threshold,
                                                filters=filters,
                                            )

                top_similarity_score = results[0].similarity if results else 0.0
                if top_similarity_score < settings.search_threshold and not should_search:
                    logger.info(f"Top similarity score {top_similarity_score:.2f} is below threshold {settings.search_threshold}.")
                    search_results = get_search_tool().search(query)
                    if search_results.success and search_results.results:
                        web_context = search_results.to_context_string()
                        logger.info(f"Web search completed. Web search returned {len(search_results.results)} results.")

                if web_context:
                    context_retriever = self._format_context(results) + "\n\n---\n\n" + web_context
                else:
                    context_retriever = self._format_context(results)
                            

            answer = chain.invoke({
                            "context": context_retriever,
                            "question": query
                        })

            logger.info(f"Answer generated")

            return {"answer": answer, "success": True}

        except Exception as e:
            logger.error(f"RAG chain failed: {e}", exc_info=True)
            return {
                "answer": "I'm having trouble answering right now. "
                          "Please try again or contact support: iordyebarnabas12@gmail.com.",
             "success": False,
             }

    def _build_filters(self, intent: IntentResult) -> Optional[RetrievalFilter]:

        """
        Creates a RetrievalFilter from the detected legal domain.

        For example, if domain is "labour", we filter to Labour Act chunks.
        This improves retrieval precision significantly.

        Args:
            intent: Classified intent with legal_domain field

        Returns:
            RetrievalFilter or None (no filtering)
        """

        domain = intent.legal_domain

        domain_to_doc = {
            "criminal":       ["2015_Administration_of_Criminal_Justice_Act", "cybercrime", 
                               "CRIMINAL PROCEDURE ACT", "CRIMINAL PROCEDURE (NORTHERN STATES) ACT", 
                               "CRIMINAL JUSTICE (MISCELLANEOUS PROVISIONS) ACT", "criminal_code_act", 
                               "criminal_procedure_act"],
            "labour":         ["labour_act"],
            "constitutional": ["constitution"],
            "electoral":      ["electoral_act"],
            "family":         ["MARRIAGE ACT", "MATRIMONIAL CAUSES ACT""CHILD'S RIGHT ACT", 
                               "BIRTHS, DEATHS, ETC. (COMPULSORY REGISTRATION) ACT"
                               ],
            "land":           ["LANDS (TITLE VESTING, ETC.) ACT", "LAND USE ACT"],
            "tax":            ["companies_income_tax_act", "personal_income_tax_act"]
        }

        if domain and domain in domain_to_doc:
            doc_keys = domain_to_doc[domain]
            # Only filter if domain is highly confident.
            # If confidence is low, search the full corpus instead.
            if intent.confidence >= 0.5:
                return RetrievalFilter(doc_keys=doc_keys)

        return None
x = Lawya()
x.answer_query("what does the law say about company tax?")