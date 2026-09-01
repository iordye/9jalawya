# src/agent/intent_detector.py
'''
INTENT TYPES:
LEGAL_QUESTION     — direct question about Nigerian law or rights
CRIMINAL_INTENT    — user expresses desire or plan to commit a crime
RIGHTS_SITUATION   — user describes a situation where their rights
                     may be violated (fired, arrested, evicted, etc.)
SOCIAL_DISTRESS    — user expresses hardship that has legal dimensions
                    (broke, homeless, domestic abuse, discrimination)
WEB_SEARCH         — user's query requires context currently not in our 
                     vector store and needs to be websearched for up to date 
                     info.                    
GENERAL_CHAT       — greetings, small talk, asking about 9jaLawya
OUT_OF_SCOPE       — completely unrelated to law or Nigerian context
 '''

#Imports
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# INTENT ENUM
class Intent(Enum):
    """
    The six possible intent categories 9jaLawya recognises.

    Using an Enum (enumeration) instead of plain strings means:
    - Typos are caught at development time ("LEGAL_QUESION" would be a
      NameError, not a silent bug)
    - IDEs provide autocomplete for valid intent values
    - Comparisons are explicit: intent == Intent.CRIMINAL_INTENT
      rather than intent == "criminal_intent" (error-prone)
    """
    LEGAL_QUESTION  = "legal_question"
    CRIMINAL_INTENT = "criminal_intent"
    RIGHTS_SITUATION = "rights_situation"
    SOCIAL_DISTRESS = "social_distress"
    GENERAL_CHAT    = "general_chat"
    OUT_OF_SCOPE    = "out_of_scope"

# INTENT RESULT DATA CLASS
@dataclass
class IntentResult:
    """
    The output of intent classification for one user message.

    Contains the detected intent, a confidence score, and extracted
    keywords that triggered the classification. The agent uses all
    three fields to decide how to respond.
    """

    intent: Intent
    confidence: float
    triggered_keywords: list
    requires_websearch: bool = False
    legal_domain: Optional[str] = None
    search_query: Optional[str] = None
    reason_websearch: Optional[str] = None
   
# KEYWORD PATTERNS
# Each category has a list of patterns.
# Patterns are checked in priority order — CRIMINAL_INTENT before
# RIGHTS_SITUATION before LEGAL_QUESTION, etc.
# The category whose patterns match the most keywords wins.

# --- Criminal Intent Patterns ---
# Words suggesting the user is considering, planning, or describing
# intent to commit a crime. We detect these to warn, not to judge.
CRIMINAL_INTENT_PATTERNS = [
    # Theft-related
    r'\b(steal|stealing|theft|rob|robbing|robbery|snatch|pickpocket|burgl)\b',
    # Violence-related
    r'\b(kill|killing|murder|attack|assault|stab|shoot|beat up|fight)\b',
    # Fraud-related
    r'\b(scam|fraud|forge|bribe|corrupt|blackmail|extort|419|yahoo)\b',
    # Drug-related
    r'\b(sell drugs|deal drugs|drug dealing|trafficking|smuggle)\b',
    #sexually related 
    r'\b(rape|force girl|harrasment|trafficking|smuggle|prostitution)\b'
    #burglary-related
    r'\b(self defense|threat|forced|entry|burglary|protection|house|)\b',
    # General criminal intent phrases
    r'\b(i want to|i feel like|i am going to|i will|planning to)\b.{0,30}'
    r'\b(steal|rob|kill|hurt|scam|cheat|fraud)\b',
    # Pidgin variants
    r'\b(i wan|i dey think|i go)\b.{0,20}\b(thief|kill|rob|scam|collect by force)\b',
    r'\b(|hack|cybercrime|digital crime|rape|sexual assault|domestic violence|human trafficking|smuggling|drug trafficking|arms dealing|statutory rape|stalking)\b'
]

# --- Rights Situation Patterns ---
# User describes a situation where their legal rights are involved.
# These are often urgent — person needs to know their rights NOW.
RIGHTS_SITUATION_PATTERNS = [
    # Employment rights
    r'\b(sack|sacked|fired|dismissed|terminate|redundan|lay.?off)\b',
    r'\b(my (boss|employer|company|oga)|they (fired|sacked|dismissed) me)\b',
    r'\b(no pay|unpaid|salary|wages|owed money)\b',
    # Arrest and detention
    r'\b(arrest|arrested|detain|detained|police|cell|lock.?up|bail)\b',
    r'\b(they arrest|dem arrest|police hold)\b',
    # Housing and property
    r'\b(evict|evicted|eviction|landlord|tenant|tenancy|rent|arrears|quit notice)\b',
    r'\b(lock.?out|force[d]? out|throw[n]? out|demolish|demolition)\b',
    r'\b(rent increase|rent dispute|security deposit|house rent)\b',
    r'\b(landlord|tenant).{0,40}\b(threaten|harass|evict|lock|enter|damage)\b',
    # Domestic and family
    r'\b(husband|wife|spouse|partner).{0,20}\b(beat|abuse|hit|threaten|force)\b',
    r'\b(domestic (violence|abuse)|sexual (assault|harassment))\b',
    # Discrimination
    r'\b(discriminat|harass|victimis|unfair treatment)\b',
    # Consumer rights
    r'\b(cheated|scammed|fake (product|goods)|refund|consumer)\b',
    r'\b(arrest|arrested|detain|detained|held|lock.?up|cell|bail|custody)\b',
    r'\b(police|officer|security).{0,40}\b(beat|hit|assault|torture|threaten|abuse|harass)\b',
    r'\b(warrant|without warrant|illegal arrest|wrongful arrest|unlawful detention)\b',
    r'\b(rights? (on|after|during) arrest|right to (lawyer|counsel|remain silent))\b',
    r'\b(search(ed)?|seize[d]?|confiscat(ed|ion))\b',
    r'\b(husband|wife|spouse|partner).{0,30}\b(beat|abuse|hit|threaten|force|rape|assault)\b',
    r'\b(domestic (violence|abuse)|gender[- ]based violence|sexual (assault|abuse|harassment))\b',
    r'\b(child abuse|child neglect|child marriage|forced marriage)\b',
    r'\b(maintenance|child support|custody|guardianship|divorce)\b',
    r'\b(forced marriage|forced to marry|marriage without consent)\b',
    r'\b(fundamental human rights|fundamental rights|constitutional rights)\b',
r'\b(right to life|right to dignity|right to privacy)\b',
r'\b(freedom of expression|freedom of speech|freedom of religion)\b',
r'\b(freedom of association|freedom of assembly|freedom of movement)\b',
r'\b(right to fair hearing|fair hearing)\b',
r'\b(school|student|teacher|university|education).{0,40}\b(expel|suspend|discriminat|harass|abuse)\b',
r'\b(education rights|student rights|school fees dispute)\b',
]

# --- Social Distress Patterns ---
# User expresses personal hardship that may have legal or social dimensions.
# 9jaLawya responds with empathy + relevant legal protections + support resources.
SOCIAL_DISTRESS_PATTERNS = [
    # Financial hardship
    r'\b(broke|no money|poor|poverty|hungry|starving|cannot afford|can\'t afford)\b',
    r'\b(i am|i\'m|i have|we are|we\'re).{0,30}\b(broke|hungry|starving|homeless|unemployed)\b'
    r'\b(no food|no job|unemployed|jobless|retrench)\b',
    # Homelessness
    r'\b(homeless|nowhere to (go|sleep|stay)|living on the street)\b',
    # Emotional distress
    r'\b(hopeless|desperate|no hope|give up|end my life|suicide)\b',
    # Family crisis
    r'\b(abandoned|my (children|kids|family) (are|is) (starving|suffering))\b',
    # Pidgin distress
    r'\b(i don finish|e don do me|i no get anything|my life don scatter)\b',
    r'\b(i no get money|i no get work|i no get food)\b',
r'\b(no be say i get money)\b',
r'\b(i dey suffer|we dey suffer)\b',
r'\b(i don tire|i don tire for everything)\b',
r'\b(e don hard|things don hard)\b',
r'\b(money no dey|food no dey)\b',
r'\b(i no know wetin to do)\b',
r'\b(i don lose everything)\b',
r'\b(my family dey suffer)\b',
]

# --- Legal Question Patterns ---
# Direct questions about law, rights, provisions, penalties, procedures.
LEGAL_QUESTION_PATTERNS = [

    # ---------------------------------------------------------
    # General legal questions
    # ---------------------------------------------------------
    r'\b(what|how|when|where|who|can|could|may|is|are|does|do).{0,50}\b'
    r'(law|legal|illegal|lawful|unlawful|right|rights|act|section|constitution|crime|offence|offense|penalty|punishment)\b',

    # "What does the law say?"
    r'\bwhat does (the|nigerian) law\b',
    r'\bwhat does the law say\b',
    r'\bwhat does the constitution say\b',

    # ---------------------------------------------------------
    # Rights
    # ---------------------------------------------------------
    r'\b(what are|what is|do i have|does? (a person|someone|i)).{0,30}\b'
    r'(rights?|entitlement|protection|freedom)\b',

    r'\b(what are my|what is my|do i have any).{0,30}\b'
    r'(rights?|legal rights?|protection)\b',

    # ---------------------------------------------------------
    # Legality
    # ---------------------------------------------------------
    r'\b(is|are|was|were|can|may).{0,40}\b'
    r'(legal|illegal|lawful|unlawful|allowed|permitted|against the law)\b',

    # ---------------------------------------------------------
    # Penalties / punishment
    # ---------------------------------------------------------
    r'\b(what is the|what are the|how much is the|how long is the).{0,30}\b'
    r'(penalty|punishment|sentence|fine|jail|prison|punishment)\b',

    r'\b(what happens if|what will happen if).{0,50}\b'
    r'(break|violate|breach|contravene|disobey)\b',

    # ---------------------------------------------------------
    # Legal procedure
    # ---------------------------------------------------------
    r'\b(how (do i|can i|would i|to)|what (is|are) the).{0,40}\b'
    r'(process|procedure|steps?|requirements?|legal process)\b',

    r'\b(where|how).{0,40}\b'
    r'(file|report|sue|appeal|complain|apply|register|challenge)\b',

    # ---------------------------------------------------------
    # Courts / lawyers
    # ---------------------------------------------------------
    r'\b(court|judge|lawyer|solicitor|barrister|legal aid).{0,40}\b'
    r'(how|where|can|should|need|do i|find|file|appeal)\b',

    # ---------------------------------------------------------
    # Nigerian legislation
    # ---------------------------------------------------------
    r'\b(labour act|criminal code|penal code|constitution|'
    r'electoral act|companies and allied matters act|CAMA|'
    r'companies act|PITA|personal income tax act|'
    r'land use act|cybercrimes act|'
    r'NDPA|Nigeria Data Protection Act)\b',

    # ---------------------------------------------------------
    # Pidgin
    # ---------------------------------------------------------
    r'\b(wetin be|wetin law talk|wetin the law say|'
    r'how e be|dem fit|person fit|i fit|'
    r'wetin my right|wetin be my right).{0,30}\b'
    r'(law|right|crime|offence|court|punishment)\b',
]

# --- General Chat Patterns ---
# Greetings, questions about 9jaLawya itself, small talk.
GENERAL_CHAT_PATTERNS = [
    r'\b(hello|hi|hey|good (morning|afternoon|evening|night)|how are you|how far|how you dey)\b',
    r'\b(who are you|what (are|can) you do|what is 9jalawya|tell me about yourself)\b',
    r'\b(thank|thanks|thank you|e go be|na wa|oya)\b',
    r'\b(ok|okay|alright|i see|i understand|noted|got it)\b',
    r'\b(help me|i need help|what can you do)\b',
]

WEBSEARCH_PATTERNS = {

    # The user explicitly asks for recent/current information.
    "current_information": r"\b(recent amendment|recently amended|latest amendment|new law|new laws|new act|new acts|recent law|recent legislation|right now|at present|presently|this week|this month|this year|recently|as of now|latest|recent|currently|current|today|yesterday|tomorrow)\b",
    # User explicitly asks for cases or court decisions.
    "case_law": r"\b(landmark case|landmark cases|precedent|precedents|judicial decision|court decision|court ruling|high court|appeal court|case law|case laws|court case|court cases|case|cases)\b.{0,50}\b(decided|judgment|judgement|ruling|decision|supreme court|court of appeal|federal high court|reported cases|recent cases)\b",
    # User explicitly asks to search/find information.
    "explicit_search": r"\b(search|search for|look up|find|find out|check|can you search|please search|search the web|look online|check online|find online|what can you find)\b",
    # User asks about changes to the law.
    "legal_changes": r"\b(amendment|amendments|amended|changed|change in the law|new legislation|new regulation|new regulations|recent legislation|recent regulation|when was.*amended|what has changed|has the law changed|did the law change)\b",
    # User asks about a specific event that may not exist
    # in the local legal corpus.
    "recent_events": r"\b(yesterday|today|last night|last week|this week|recently|just happened|just announced|just released|current court proceedings|ongoing case|ongoing trial|current proceedings|ongoing proceedings)\b",
    # User explicitly asks about information available online.
    "online_information": r"\b(online|on the internet|on the web|web search|internet sources|online sources|according to recent reports|news reports|news report)\b",
}


# --- Legal Domain Sub-classification ---
# Used to set the legal_domain field when intent is LEGAL_QUESTION
# or RIGHTS_SITUATION, enabling smart retrieval filters.
DOMAIN_PATTERNS = {
    "criminal":      r'\b(crime|criminal|offence|theft|murder|fraud|assault|sentence|prison|police|arrest|hack|cybercrime|digital crime|rape|sexual assault|domestic violence|human trafficking|smuggling|drug trafficking|arms dealing|statutory rape|stalking)\b',
    "labour":        r'\b(employ|work|job|salary|wages|dismiss|fire|sack|overtime|leave|contract of service)\b',
    "constitutional":r'\b(right|freedom|constitution|fundamental|citizen|government|president|governor|state)\b',
    "electoral":     r'\b(election|vote|voting|inec|candidate|party|ballot|result|rigging)\b',
    "tax":           r'\b(tax|income tax|CITA|PITA|FIRS|VAT|assessment|revenue|PAYE|company tax)\b',
    "family":        r'\b(marriage|divorce|child|custody|inheritance|will|estate|domestic|mother|father|son|daughter|parent)\b',
    "land":          r'\b(land|property|title|deed|boundary|survey|estate|ownership|landlord|tenant)\b',
}

# --- Search Query Templates ---
# When certain intents are detected, we construct a better RAG query
# than the raw user message. These templates use {keyword} placeholders
# that are filled from the triggered_keywords list.
SEARCH_QUERY_TEMPLATES = {
    Intent.CRIMINAL_INTENT: "punishment penalty {keyword} Nigeria criminal code",
    Intent.RIGHTS_SITUATION: "{keyword} rights Nigeria law protection",
    Intent.SOCIAL_DISTRESS: "government assistance poverty relief Nigeria law",
    Intent.LEGAL_QUESTION: "{query}",  # Use original query for direct questions
}

# INTENT DETECTOR CLASS
class IntentDetector:
    """
    Classifies user messages into intent categories for 9jaLawya.

    Approach: cascading keyword pattern matching with priority ordering.
    CRIMINAL_INTENT is checked first because it requires the most urgent
    and specific response. GENERAL_CHAT is checked last as a catch-all.

    The detector is intentionally permissive — it prefers false positives
    (treating a casual comment as legally relevant) over false negatives
    (missing a real legal situation). A user who says "I feel like killing
    my boss" probably means it figuratively, but 9jaLawya should still
    gently mention what the law says — that is the whole point of the tool.
    """

    def __init__(self):
        # Pre-compile all regex patterns once at initialisation.
        # Compiled patterns are reused across all .detect() calls —
        # much faster than re-compiling on every message.
        self._criminal_patterns = [
            re.compile(p, re.IGNORECASE) for p in CRIMINAL_INTENT_PATTERNS
        ]
        self._rights_patterns = [
            re.compile(p, re.IGNORECASE) for p in RIGHTS_SITUATION_PATTERNS
        ]
        self._distress_patterns = [
            re.compile(p, re.IGNORECASE) for p in SOCIAL_DISTRESS_PATTERNS
        ]
        self._legal_patterns = [
            re.compile(p, re.IGNORECASE) for p in LEGAL_QUESTION_PATTERNS
        ]
        self._chat_patterns = [
            re.compile(p, re.IGNORECASE) for p in GENERAL_CHAT_PATTERNS
        ]
        self._websearch_patterns = {
            web_pattern: re.compile(pattern, re.IGNORECASE)
            for web_pattern, pattern in WEBSEARCH_PATTERNS.items()
        }
        self._domain_patterns = {
            domain: re.compile(pattern, re.IGNORECASE)
            for domain, pattern in DOMAIN_PATTERNS.items()
        }
        # re.IGNORECASE makes patterns case-insensitive so "Steal",
        # "STEAL", and "steal" all match the same pattern.
        

        logger.info("IntentDetector initialised with compiled patterns.")

    def detect(self, message: str) -> IntentResult:
        """
        Classifies a user message into an intent category.

        Checks patterns in priority order:
        1. CRIMINAL_INTENT  — highest priority, needs immediate response
        2. RIGHTS_SITUATION — user may need urgent legal help
        3. SOCIAL_DISTRESS  — user needs empathy + resources
        4. LEGAL_QUESTION   — direct legal query
        5. GENERAL_CHAT     — conversational
        6. OUT_OF_SCOPE     — fallback

        Args:
            message: Raw user message string

        Returns:
            IntentResult with intent, confidence, keywords, domain, query
        """

        if not message or not message.strip():
            return IntentResult(
                intent=Intent.GENERAL_CHAT,
                confidence=1.0,
                triggered_keywords=[],
            )

        msg = message.strip()

        # Priority 1: Criminal Intent
        criminal_keywords = self._match_keywords(msg, self._criminal_patterns)
        if criminal_keywords:
            confidence = min(0.6 + 0.1 * len(criminal_keywords), 1.0)
            # Base confidence 0.6 + 0.1 per additional keyword, capped at 1.0
            # More matching keywords = stronger signal of criminal intent.
            primary_keyword = criminal_keywords[0]
            web_search, why = self._requires_websearch(msg)
            return IntentResult(
                intent=Intent.CRIMINAL_INTENT,
                confidence=confidence,
                triggered_keywords=criminal_keywords,
                legal_domain="criminal",
                search_query=f"punishment penalty {primary_keyword} Nigeria criminal code",
                requires_websearch=web_search,
                reason_websearch=why 
            )

        # Priority 2: Rights Situation
        rights_keywords = self._match_keywords(msg, self._rights_patterns)
        if rights_keywords:
            confidence = min(0.65 + 0.08 * len(rights_keywords), 1.0)
            domain = self._detect_domain(msg)
            web_search, why = self._requires_websearch(msg)
            primary_keyword = rights_keywords[0]
            return IntentResult(
                intent=Intent.RIGHTS_SITUATION,
                confidence=confidence,
                triggered_keywords=rights_keywords,
                legal_domain=domain or "constitutional",
                search_query=f"{primary_keyword} rights Nigeria law protection",
                requires_websearch = web_search,
                reason_websearch=why 
            )

        # Priority 3: Social Distress
        distress_keywords = self._match_keywords(msg, self._distress_patterns)
        if distress_keywords:
            # Check if there is also a criminal or legal dimension to the distress.
            # "I am broke and I want to steal" has both distress AND criminal intent.
            # We already caught criminal intent in Priority 1, so if we are here,
            # the distress is the primary signal — but we still check for legal dims.
            confidence = min(0.6 + 0.08 * len(distress_keywords), 1.0)
            web_search, why = self._requires_websearch(msg)
            return IntentResult(
                intent=Intent.SOCIAL_DISTRESS,
                confidence=confidence,
                triggered_keywords=distress_keywords,
                search_query="government assistance poverty support Nigeria",
                requires_websearch = web_search,
                reason_websearch=why 
            )

        # --- Priority 4: Legal Question ---
        legal_keywords = self._match_keywords(msg, self._legal_patterns)
        if legal_keywords:
            confidence = min(0.7 + 0.05 * len(legal_keywords), 1.0)
            domain = self._detect_domain(msg)
            web_search, why = self._requires_websearch(msg)
            return IntentResult(
                intent=Intent.LEGAL_QUESTION,
                confidence=confidence,
                triggered_keywords=legal_keywords,
                legal_domain=domain,
                search_query=msg,
                requires_websearch = web_search,
                reason_websearch=why 
                # For direct legal questions, the original message is the
                # best RAG query — it already uses legal terminology.
            )

        # --- Priority 5: General Chat ---
        chat_keywords = self._match_keywords(msg, self._chat_patterns)
        if chat_keywords:
            web_search, why = self._requires_websearch(msg)
            return IntentResult(
                intent=Intent.GENERAL_CHAT,
                confidence=0.8,
                triggered_keywords=chat_keywords,
                requires_websearch = web_search,
                reason_websearch=why 
            )

        # --- Priority 6: Fallback ---
        # Message matched no patterns. Could be:
        # - An implicit legal situation without trigger words
        # - A topic 9jaLawya doesn't cover
        # - Pidgin or informal phrasing we haven't covered yet
        # We default to LEGAL_QUESTION with low confidence so the agent
        # still attempts a RAG retrieval — better to try and miss than
        # to refuse a valid legal question because it used unusual words.
        logger.debug(f"No pattern matched for message: '{msg[:60]}'. Defaulting to LEGAL_QUESTION.")
        web_search, why = self._requires_websearch(msg)
        return IntentResult(
            intent=Intent.LEGAL_QUESTION,
            confidence=0.3,
            triggered_keywords=[],
            search_query=msg,
            requires_websearch = web_search,
            reason_websearch=why 
        )

    def _match_keywords(self, message: str, patterns: list) -> list:
        """
        Finds all keyword matches in the message for a given pattern list.

        Args:
            message: User message string
            patterns: List of compiled regex patterns

        Returns:
            List of matched keyword strings (deduplicated)
        """
        matched = []
        for pattern in patterns:
            found = pattern.findall(message)
            # .findall() returns a list of all non-overlapping matches.
            # For groups in the pattern, it returns the group content.
            for match in found:
                if isinstance(match, tuple):
                    # Some patterns use groups — findall returns tuples.
                    # We take the first non-empty group.
                    match = next((m for m in match if m), "")
                if match and match.lower() not in [k.lower() for k in matched]:
                    matched.append(match.strip())
                    # Deduplicate case-insensitively so "Steal" and "steal"
                    # don't both appear in the keywords list.
        return matched

    def _detect_domain(self, message: str) -> Optional[str]:
        """
        Identifies the specific legal domain of a message.

        Checks the message against domain-specific patterns and returns
        the domain with the most keyword matches. Used to set
        IntentResult.legal_domain for smart retrieval filtering.

        Args:
            message: User message string

        Returns:
            Domain string ("criminal", "labour", etc.) or None
        """
        domain_scores = {}
        for domain, pattern in self._domain_patterns.items():
            matches = pattern.findall(message)
            if matches:
                domain_scores[domain] = len(matches)

        if not domain_scores:
            return None

        # Return the domain with the highest match count.
        return max(domain_scores, key=domain_scores.get)
        # max() with key= returns the item that maximises the key function.
        # domain_scores.get returns the count for each domain key.
    
    def _requires_websearch(self, message: str) -> bool | tuple[bool, Optional[str]]:
        
        web_scores = {}
        for i, pattern in self._websearch_patterns.items():
            matches = pattern.findall(message)
            if matches:
                web_scores[i] = len(matches)

        if not web_scores:
            return False, None
        
        return True, max(web_scores, key=web_scores.get)

# MODULE-LEVEL CONVENIENCE FUNCTION
# Single shared instance — created once, reused across all calls.
# This avoids re-compiling all regex patterns on every message.
_detector_instance: Optional[IntentDetector] = None

def detect_intent(message: str) -> IntentResult:
    """
    Module-level convenience function for intent detection.

    Creates the IntentDetector singleton on first call, then reuses it.
    This is the function that law_buddy.py will call — it never needs
    to instantiate IntentDetector directly.

    Args:
        message: User message string

    Returns:
        IntentResult with classification
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = IntentDetector()
    return _detector_instance.detect(message)

# ENTRY POINT — for testing
if __name__ == "__main__":
    print("\n9jaLawya Intent Detector — Test Mode")
    print("="*50)
    print("Type any message to see how it is classified.")
    print("Type 'quit' to exit.\n")

    detector = IntentDetector()

    test_messages = [
        "I feel like stealing because I am broke",
        "My landlord locked me out of my house",
        "What does the law say about employee termination?",
        "I am so hungry and I have no money",
        "Hello, how are you?",
        "My boss has not paid my salary in 3 months",
        "wetin be the punishment for robbery for Nigeria?",
        "i wan rob somebody make i get money",
        "dem arrest my brother for no reason",
        "What are my rights if I am arrested?",
    ]

    print("Running test messages:\n")
    for msg in test_messages:
        result = detector.detect(msg)
        print(f"Message:   {msg}")
        print(f"Intent:    {result.intent.value}")
        print(f"Confidence:{result.confidence:.2f}")
        print(f"Keywords:  {result.triggered_keywords}")
        print(f"Domain:    {result.legal_domain}")
        print(f"Query:     {result.search_query}")
        print("-" * 50)

    print("\nInteractive mode:")
    while True:
        try:
            user_input = input("\nMessage > ").strip()
            if user_input.lower() == "quit":
                break
            if not user_input:
                continue
            result = detector.detect(user_input)
            print(f"  Intent:     {result.intent.value}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Keywords:   {result.triggered_keywords}")
            print(f"  Domain:     {result.legal_domain}")
            print(f"  RAG Query:  {result.search_query}")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nDone.")
