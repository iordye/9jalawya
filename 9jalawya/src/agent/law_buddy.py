# src/agent/law_buddy.py
''' THE 9jaLawya PERSONALITY:
 9jaLawya speaks like a knowledgeable Nigerian friend — not a judge,
 not a textbook, not a formal legal database. It:
   - Uses warm, accessible language (can mix in light Pidgin naturally)
   - Never lectures or moralises
   - Always grounds answers in actual Nigerian law with citations
   - Expresses genuine empathy for difficult situations
   - Proactively mentions relevant rights the user may not have asked about
   - Points to real support resources, not just legal text'''

# Imports
import sys
import time
import logging
import torch
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.retriever import NaijaLawRetriever, RetrievalFilter
from agent.intent_detector import detect_intent, Intent, IntentResult
from agent.web_search import get_search_tool, WebSearchResponse

# Logging setup 
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# CONSTANTS
GENERATION_MODEL = str(PROJECT_ROOT / "models" / "qwen2-0.5b-instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512
# Maximum tokens TinyLlama generates per response.
# 512 tokens ≈ 380 words — enough for a thorough friendly legal explanation.
TEMPERATURE = 0.7
MAX_CONTEXT_CHUNKS = 3
CONVERSATION_MEMORY_TURNS = 6
DISTRESS_KEYWORD_MAP = {
    "broke": "poverty", "no money": "poverty", "poor": "poverty",
    "hungry": "food", "no food": "food", "starving": "food",
    "homeless": "homeless", "nowhere to go": "homeless",
    "unemployed": "unemployment", "no job": "unemployment",
    "domestic": "domestic_violence", "abuse": "domestic_violence",
    "beat": "domestic_violence", "husband": "domestic_violence",
    "arrested": "arrest", "police": "arrest",
    "discriminat": "discrimination",
}

# CONVERSATION MEMORY
@dataclass
class ConversationTurn:
    """A single exchange in the conversation."""
    role: str        # "user" or "assistant"
    content: str     # The message text

class ConversationMemory:
    """
    Stores the recent conversation history for context.

    TinyLlama is a stateless model — it has no memory between calls.
    We give it memory by including recent exchanges in every prompt.
    This allows follow-up questions like "tell me more about that" to work.
    """

    def __init__(self, max_turns: int = CONVERSATION_MEMORY_TURNS):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
        # max_turns is the number of complete user+assistant pairs to keep.

    def add(self, role: str, content: str):
        """Adds a new turn to memory, dropping oldest if over limit."""
        self.turns.append(ConversationTurn(role=role, content=content))
        # Keep only the last max_turns * 2 individual messages
        # (each turn = 1 user + 1 assistant message = 2 entries).
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]
            # Slice from the end: keep the last N*2 entries.

    def get_history(self) -> List[ConversationTurn]:
        """Returns all stored turns."""
        return self.turns

    def clear(self):
        """Clears all conversation history."""
        self.turns = []

# PROMPT BUILDER
class PromptBuilder:
    """
    Constructs prompts using the tokenizer's built-in chat template.
    """

    # The 9jaLawya system prompt — defines personality and behaviour.
    # This is injected once at the start of every conversation.
    # The content stays the same regardless of model — only the FORMAT
    # wrapper (how the system prompt is marked up) changes per model,
    # and apply_chat_template() handles that automatically.
    SYSTEM_PROMPT = """You are 9jaLawya, a friendly and knowledgeable Nigerian legal assistant. You speak like a trusted friend who knows Nigerian law inside out.

Your personality:
- Warm, approachable, and genuinely caring — never cold or robotic
- You can mix in light Nigerian Pidgin naturally when it fits the conversation
- You never judge or lecture users, even when they express difficult things
- You always ground your answers in actual Nigerian law and cite the specific section and act
- You proactively mention rights and protections users may not know about
- You point users to real help when they are in distress

Your approach:
- When someone expresses intent to commit a crime: respond with empathy first, explain what the law says about the consequences in a friendly way, and suggest alternatives or support
- When someone describes a rights violation: clearly explain their legal rights, cite the law, and tell them what steps they can take
- When someone is in distress: acknowledge their pain first, then share relevant legal protections and support resources
- When asked a legal question: give a clear, cited answer that a non-lawyer can actually understand
- Always cite the specific section and act name (e.g. "Section 9 of the Labour Act")

Remember: You are not giving formal legal advice — you are helping people understand their rights and the law. Always encourage them to consult a qualified lawyer for serious matters."""

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: The loaded HuggingFace tokenizer for the current model.
                       We store it so apply_chat_template() can be called
                       on every prompt build.
        """
        self.tokenizer = tokenizer

        # Verify the tokenizer has a chat template.
        # Most modern chat models do. If not, we fall back to a simple format.
        self._has_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        # hasattr() checks if the attribute exists on the object at all.
        # We check both existence AND non-None because some tokenizers
        # have the attribute but set it to None (no template defined).

        if self._has_chat_template:
            logger.info("  Prompt builder: using tokenizer.apply_chat_template()")
        else:
            logger.warning(
                "  Prompt builder: tokenizer has no chat_template. "
                "Falling back to simple format. Response quality may be reduced."
            )

    def build_prompt(
        self,
        user_message: str,
        legal_context: str,
        web_context: str,
        conversation_history,
        intent,
    ) -> str:
        """
        Builds the complete prompt for one response using the tokenizer's
        own chat template.

        Constructs a messages list in the standard OpenAI-style format:
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "user",      "content": "..."},  ← current message
        ]

        Then passes it to apply_chat_template() which formats it correctly
        for whichever model is loaded.

        Args:
            user_message: The current user input
            legal_context: Formatted string of retrieved legal chunks
            web_context: Formatted string of web search results (may be empty)
            conversation_history: Previous ConversationTurn objects
            intent: Classified IntentResult for the current message

        Returns:
            Complete prompt string formatted for the current model
        """

        # Build the context injection for the current user turn.
        context_block = self._build_context_block(legal_context, web_context, intent)

        # Attach context to the user message if we have any.
        if context_block:
            current_user_content = (
                f"{context_block}\n\n"
                f"User message: {user_message}"
            )
        else:
            current_user_content = user_message

        # Assemble the messages list.
        # This is the universal format all HuggingFace chat models accept
        # via apply_chat_template().
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            # System message always comes first and is injected once.
            # apply_chat_template() knows how to wrap this correctly for
            # each model — TinyLlama uses <<SYS>>, Qwen2 uses <|im_start|>system
        ]

        # Inject conversation history as alternating user/assistant messages.
        for turn in conversation_history:
            messages.append({
                "role": turn.role,
                # turn.role is "user" or "assistant"
                "content": turn.content,
            })

        # Add the current user message (with context attached).
        messages.append({
            "role": "user",
            "content": current_user_content,
        })

        if self._has_chat_template:
            # Use the model's own chat template.
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                # tokenize=False: return a string, not token IDs.
                # We want the string so we can log it and pass it to
                # the tokenizer ourselves with our own settings.
                add_generation_prompt=True,
                # add_generation_prompt=True: appends the model-specific
                # token(s) that signal "now generate the assistant response".
                # For TinyLlama this adds " [/INST]"
                # For Qwen2 this adds "<|im_start|>assistant\n"
                # Without this, the model does not know it should start
                # generating — it might instead continue the user turn.
            )
        else:
            # Fallback: simple format without model-specific markers.
            # Less effective but works as a safety net.
            prompt = self._build_fallback_prompt(messages)

        return prompt

    def _build_context_block(
        self,
        legal_context: str,
        web_context: str,
        intent,
    ) -> str:
        """
        Assembles the context injection block from legal and web sources.

        Args:
            legal_context: Retrieved legal chunks as formatted string
            web_context: Web search results as formatted string
            intent: Used to write an appropriate context header

        Returns:
            Formatted context block string, or empty string if no context
        """
        from agent.intent_detector import Intent

        parts = []

        if legal_context and "No relevant" not in legal_context:
            parts.append(f"RELEVANT NIGERIAN LAW:\n{legal_context}")

        if web_context and "No relevant" not in web_context:
            parts.append(f"SUPPORT RESOURCES (from web):\n{web_context}")

        if not parts:
            return ""

        header = self._get_context_header(intent)
        return f"{header}\n\n" + "\n\n---\n\n".join(parts)

    def _get_context_header(self, intent) -> str:
        """Returns an appropriate instruction header based on intent."""
        from agent.intent_detector import Intent

        headers = {
            Intent.CRIMINAL_INTENT:    "Use the following Nigerian law to explain the legal consequences in a friendly, non-judgmental way, and suggest support options:",
            Intent.RIGHTS_SITUATION:   "Use the following Nigerian law to explain the user's rights clearly and what steps they can take:",
            Intent.SOCIAL_DISTRESS:    "Use the following information to help this person with empathy, their legal protections, and available support:",
            Intent.LEGAL_QUESTION:     "Answer the user's question using the following Nigerian legal provisions, citing the specific section and act:",
            Intent.GENERAL_CHAT:       "",
            Intent.OUT_OF_SCOPE:       "",
        }
        return headers.get(intent.intent, "Use the following context to help the user:")

    def _build_fallback_prompt(self, messages: list) -> str:
        """
        Simple fallback prompt format when no chat template is available.

        Args:
            messages: List of role/content dicts

        Returns:
            Simple formatted prompt string
        """
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            if role == "SYSTEM":
                parts.append(f"Instructions: {content}")
            elif role == "USER":
                parts.append(f"User: {content}")
            elif role == "ASSISTANT":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        # "Assistant:" at the end signals the model to generate a response.
        return "\n\n".join(parts)

# 9jaLAWYA AGENT — MAIN CLASS
class NijaLawyaAgent:
    """
    The main 9jaLawya conversational agent.

    Orchestrates intent detection, retrieval, web search, and generation
    to produce warm, legally-grounded responses to any user message.

    Usage:
        agent = NijaLawyaAgent()
        response = agent.chat("I feel like stealing because I am broke")
        print(response)
    """

    def __init__(self):
        """
        Loads all components. This takes 30-60 seconds on first run
        due to model downloads. Subsequent runs are faster (cache).
        """
        logger.info("="*60)
        logger.info("Initialising 9jaLawya...")
        logger.info("="*60)

        # --- Load retriever ---
        logger.info("Loading retriever and vector store...")
        self.retriever = NaijaLawRetriever()

        # --- Load web search tool ---
        logger.info("Initialising web search tool...")
        self.search_tool = get_search_tool()

        # --- Load TinyLlama ---
        logger.info(f"Loading TinyLlama ({GENERATION_MODEL})...")
        logger.info("First run downloads ~2.2GB — please wait...")
        logger.info(f"Running on: {DEVICE.upper()}")

        model_start = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
            # AutoTokenizer detects the right tokenizer class for TinyLlama
            # and loads vocabulary + special token definitions.

            self.model = AutoModelForCausalLM.from_pretrained(
                GENERATION_MODEL,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                # low_cpu_mem_usage=True: load model weights progressively
                # rather than all at once. Prevents RAM spike during loading.
            )
            self.model = self.model.to(DEVICE)
            self.model.eval()
            # eval() disables dropout — essential for consistent inference.

        except Exception as e:
            logger.error(f"Failed to load TinyLlama: {e}")
            raise

        model_elapsed = round(time.time() - model_start, 1)
        logger.info(f"TinyLlama loaded in {model_elapsed}s")

        # --- Set up components ---
        self.prompt_builder = PromptBuilder(tokenizer=self.tokenizer)
        self.memory = ConversationMemory()

        logger.info("="*60)
        logger.info("9jaLawya is ready! E don ready!")
        logger.info("="*60)

    def chat(self, user_message: str, stream: bool = True) -> str:
        """
        The main method — processes a user message and returns a response.

        Full pipeline:
        1. Detect intent of the message
        2. Retrieve relevant legal chunks (always)
        3. Fetch web resources (if distress or rights situation)
        4. Build the TinyLlama prompt
        5. Generate the response
        6. Update conversation memory

        Args:
            user_message: Raw user input string
            stream: If True, print tokens as generated (more responsive feel)

        Returns:
            Generated response string
        """

        if not user_message or not user_message.strip():
            return "Abeg, type something Weh I help you with! 😊"

        user_message = user_message.strip()
        logger.info(f"\nUser: {user_message[:80]}")

        # ── STEP 1: DETECT INTENT ───────────────────────────────────────────
        intent = detect_intent(user_message)
        logger.info(
            f"Intent: {intent.intent.value} "
            f"(confidence: {intent.confidence:.2f}, "
            f"domain: {intent.legal_domain})"
        )

        # ── STEP 2: RETRIEVE LEGAL CONTEXT ──────────────────────────────────
        legal_context = ""

        if intent.intent != Intent.GENERAL_CHAT:
            # Build retrieval filters from detected domain.
            filters = self._build_filters(intent)

            # Use the intent's suggested search query, not the raw message.
            # This bridges casual/Pidgin input to formal legal terminology.
            retrieval_query = intent.search_query or user_message

            results = self.retriever.retrieve(
                query=retrieval_query,
                top_k=MAX_CONTEXT_CHUNKS,
                filters=filters,
            )

            if results:
                # Format top chunks as context string.
                context_blocks = [r.to_context_string() for r in results[:MAX_CONTEXT_CHUNKS]]
                legal_context = "\n\n---\n\n".join(context_blocks)
                logger.info(f"Retrieved {len(results)} legal chunks.")
            else:
                logger.info("No legal chunks retrieved above threshold.")

        # ── STEP 3: WEB SEARCH (if needed) ──────────────────────────────────
        web_context = ""

        should_search = intent.intent in (
            Intent.SOCIAL_DISTRESS,
            Intent.CRIMINAL_INTENT,
            Intent.RIGHTS_SITUATION,
        )

        if should_search:
            distress_type = self._map_distress_type(intent)
            logger.info(f"Running web search for: {distress_type}")
            web_response = self.search_tool.search_for_resources(distress_type)
            if web_response.success and web_response.results:
                web_context = web_response.to_context_string()
                logger.info(f"Web search returned {len(web_response.results)} results.")

        # ── STEP 4: BUILD PROMPT ────────────────────────────────────────────
        prompt = self.prompt_builder.build_prompt(
            user_message=user_message,
            legal_context=legal_context,
            web_context=web_context,
            conversation_history=self.memory.get_history(),
            intent=intent,
        )

        # ── STEP 5: GENERATE RESPONSE ────────────────────────────────────────
        logger.info("Generating response...")
        generation_start = time.time()

        response = self._generate(prompt, stream=stream)

        generation_elapsed = round(time.time() - generation_start, 1)
        logger.info(f"Generation complete in {generation_elapsed}s")

        # ── STEP 6: UPDATE MEMORY ────────────────────────────────────────────
        self.memory.add("user", user_message)
        self.memory.add("assistant", response)

        return response

    def _generate(self, prompt: str, stream: bool = True) -> str:
        """
        Runs TinyLlama inference to generate a response.

        Args:
            prompt: Complete formatted prompt string
            stream: Whether to stream tokens to stdout as generated

        Returns:
            Generated text string
        """

        # Tokenise the prompt.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            # TinyLlama's context window is 2048 tokens.
            # truncation=True ensures we never exceed this limit.
        ).to(DEVICE)

        streamer = None
        if stream:
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                # skip_prompt=True: only stream the newly generated tokens,
                # not the input prompt we fed in.
                skip_special_tokens=True,
                # skip_special_tokens=True: don't print <s>, </s>, [INST] etc.
            )
            print("\n9jaLawya: ", end="", flush=True)
            # Print the speaker label before streaming begins.

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                # do_sample=True: use sampling (temperature-based) rather
                # than greedy decoding. Required when temperature != 0.
                # Sampling produces more natural, varied responses.
                top_p=0.9,
                # top_p (nucleus sampling): at each step, only consider tokens
                # whose cumulative probability reaches 0.9. Filters out
                # very unlikely tokens while preserving natural variety.
                # Works with do_sample=True.
                repetition_penalty=1.15,
                # repetition_penalty > 1.0 penalises repeating the same
                # tokens. 1.15 prevents the model from getting stuck in
                # loops like "the law says the law says the law says..."
                # Common failure mode in small models.
                pad_token_id=self.tokenizer.eos_token_id,
                # pad_token_id: tells the model which token signals end of
                # generation. Without this, some models generate indefinitely.
                streamer=streamer,
            )

        # Decode only the newly generated tokens (not the input prompt).
        input_length = inputs["input_ids"].shape[1]
        # .shape[1] gives the number of input tokens.

        new_tokens = output_ids[0][input_length:]
        # output_ids[0] is the first (only) generated sequence.
        # [input_length:] slices off the input tokens, leaving only
        # the newly generated response tokens.

        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.strip()

        if stream:
            print()
            # Print a newline after streaming completes.

        return response

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
            "criminal":       ["criminal_code_act"],
            "labour":         ["labour_act"],
            "constitutional": ["constitution"],
            "electoral":      ["electoral_act"],
            "tax":            ["companies_income_tax_act", "personal_income_tax_act"],
        }

        if domain and domain in domain_to_doc:
            doc_keys = domain_to_doc[domain]
            # Only filter if domain is highly confident.
            # If confidence is low, search the full corpus instead.
            if intent.confidence >= 0.5:
                return RetrievalFilter(doc_keys=doc_keys)

        return None
        # None means no filtering — search entire corpus.

    def _map_distress_type(self, intent: IntentResult) -> str:
        """
        Maps the intent's triggered keywords to a web search resource type.

        Args:
            intent: Classified intent with triggered_keywords

        Returns:
            Resource type string (key in RESOURCE_QUERIES)
        """

        for keyword in intent.triggered_keywords:
            keyword_lower = keyword.lower()
            for signal, resource_type in DISTRESS_KEYWORD_MAP.items():
                if signal in keyword_lower:
                    return resource_type

        # Default fallbacks based on intent type.
        if intent.intent == Intent.CRIMINAL_INTENT:
            return "legal_aid"
        elif intent.intent == Intent.RIGHTS_SITUATION:
            return "legal_aid"
        else:
            return "general_support"

    def clear_memory(self):
        """Clears conversation history. Useful for starting a new session."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")


# -----------------------------------------------------------------------------
# ENTRY POINT — for testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("9jaLawya — Your Nigerian Law Buddy")
    print("="*60)
    print("Chat with 9jaLawya about any legal matter or situation.")
    print("Commands: 'clear' (reset memory) | 'quit' (exit)")
    print("="*60 + "\n")

    try:
        agent = NijaLawyaAgent()
    except Exception as e:
        logger.error(f"Failed to start 9jaLawya: {e}")
        sys.exit(1)

    print("\n9jaLawya is ready! Ask me anything about Nigerian law.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nE don do! Take care! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("9jaLawya: E don do! Stay safe and know your rights! 👋")
            break

        if user_input.lower() == "clear":
            agent.clear_memory()
            print("9jaLawya: Memory cleared! We dey start fresh. 😊\n")
            continue

        # Generate and stream the response.
        agent.chat(user_input, stream=True)
        print()
