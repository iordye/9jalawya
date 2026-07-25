# 9jalawya/main.py

# Imports 
import sys
import argparse
import logging
from pathlib import Path

# PATH SETUP
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ARGUMENT PARSING
def parse_args():
    """
    Parses command-line arguments passed when running main.py.

    argparse is Python's built-in argument parsing library.
    It automatically generates --help output and validates argument types.

    Returns:
        Namespace object with all parsed arguments as attributes.
        e.g. args.no_stream, args.debug, args.no_web
    """
    parser = argparse.ArgumentParser(
        description="9jaLawya — Your Nigerian Law Buddy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # RawDescriptionHelpFormatter preserves line breaks in the epilog.
        epilog=(
            "Examples:\n"
            "  python main.py                  # Normal mode\n"
            "  python main.py --no-stream      # Print full answers at once\n"
            "  python main.py --no-web         # Offline mode (no web search)\n"
            "  python main.py --debug          # Verbose logging\n"
        )
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        # action="store_true": flag is False by default, True if passed.
        # So: python main.py           → args.no_stream = False (streaming ON)
        #     python main.py --no-stream → args.no_stream = True (streaming OFF)
        default=False,
        help="Disable token streaming. Print full response at once instead.",
    )

    parser.add_argument(
        "--no-web",
        action="store_true",
        default=False,
        help="Disable web search. Run in offline mode using only the vector store.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG level logging. Shows detailed internal pipeline steps.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------

def setup_logging(debug: bool = False):
    """
    Configures the logging level for the entire application.

    In normal mode (debug=False): only INFO, WARNING, ERROR appear.
    In debug mode (debug=True):   all DEBUG messages appear too —
    useful for seeing exactly what intent was detected, what chunks
    were retrieved, how many tokens the prompt used, etc.

    Args:
        debug: Whether to enable DEBUG level logging
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        # force=True: overrides any previous basicConfig calls from
        # imported modules (retriever.py, law_buddy.py etc. all call
        # basicConfig themselves — force=True ensures our settings win).
    )


# -----------------------------------------------------------------------------
# ENVIRONMENT VALIDATION
# -----------------------------------------------------------------------------

def validate_environment() -> bool:
    """
    Checks that all required components are in place before starting.

    Catches common setup mistakes early with clear error messages rather
    than letting the user wait 60 seconds for TinyLlama to load only to
    crash because the vector store doesn't exist.

    Checks:
    1. Required Python packages are installed
    2. Vector store exists and is not empty
    3. src/ directory structure is correct

    Returns:
        True if all checks pass, False if any critical check fails.
    """

    all_ok = True

    print("Checking environment...")

    # --- Check 1: Required packages ---
    required_packages = {
        "chromadb":             "pip install chromadb",
        "sentence_transformers":"pip install sentence-transformers",
        "transformers":         "pip install transformers",
        "torch":                "pip install torch",
        "pdfplumber":           "pip install pdfplumber",
        "tqdm":                 "pip install tqdm",
    }

    for package, install_cmd in required_packages.items():
        try:
            __import__(package)
            # __import__() dynamically imports a module by name string.
            # If it succeeds, the package is installed.
            # If it raises ImportError, the package is missing.
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not found — run: {install_cmd}")
            all_ok = False

    # Optional package — warn but don't fail.
    try:
        __import__("duckduckgo_search")
        print(f"  ✓ duckduckgo_search (web search enabled)")
    except ImportError:
        print(
            f"  ⚠ duckduckgo_search not found — web search disabled.\n"
            f"    Install with: pip install duckduckgo-search"
        )
        # Not setting all_ok = False — web search is optional.

    # --- Check 2: Vector store ---
    vectorstore_path = PROJECT_ROOT / "vectorstore"
    if not vectorstore_path.exists():
        print(
            f"\n  ✗ Vector store not found at: {vectorstore_path}\n"
            f"    Run the full ingestion pipeline first:\n"
            f"      python src/ingestion/pdf_extractor.py\n"
            f"      python src/ingestion/cleaner.py\n"
            f"      python src/ingestion/chunker.py\n"
            f"      python src/ingestion/ingestor.py"
        )
        all_ok = False
    else:
        # Check if vectorstore has actual content (not just an empty folder).
        chroma_files = list(vectorstore_path.rglob("*.sqlite3")) + \
                       list(vectorstore_path.rglob("chroma.sqlite3"))
        # rglob("*.sqlite3") searches recursively for ChromaDB's SQLite file.
        # Its presence confirms the database was actually populated.
        if chroma_files:
            print(f"  ✓ Vector store found at: {vectorstore_path}")
        else:
            # Folder exists but no database file — ingestion was not completed.
            print(
                f"  ⚠ Vector store folder exists but appears empty.\n"
                f"    Run: python src/ingestion/ingestor.py"
            )
            # Warning not hard failure — ChromaDB might use different filenames.

    # --- Check 3: Source directory structure ---
    required_dirs = [
        PROJECT_ROOT / "src" / "ingestion",
        PROJECT_ROOT / "src" / "retrieval",
        PROJECT_ROOT / "src" / "agent",
    ]

    for d in required_dirs:
        if d.exists():
            print(f"  ✓ {d.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  ✗ Missing directory: {d.relative_to(PROJECT_ROOT)}")
            all_ok = False

    print()
    return all_ok


# -----------------------------------------------------------------------------
# WELCOME BANNER
# -----------------------------------------------------------------------------

def print_banner():
    """
    Prints the 9jaLawya welcome banner when the app starts.

    A clear banner:
    1. Confirms the app started successfully
    2. Sets the tone (friendly, Nigerian identity)
    3. Tells users what they can do and what commands exist
    """

    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        ██████╗      ██╗ █████╗    ██╗      █████╗           ║
║       ██╔═══██╗     ██║██╔══██╗   ██║     ██╔══██╗          ║
║       ██║   ██║     ██║███████║   ██║     ███████║          ║
║       ██║▄▄ ██║██   ██║██╔══██║   ██║     ██╔══██║          ║
║       ╚██████╔╝╚█████╔╝██║  ██║   ███████╗██║  ██║          ║
║        ╚══▀▀═╝  ╚════╝ ╚═╝  ╚═╝   ╚══════╝╚═╝  ╚═╝          ║
║                                                              ║
║              W Y A   —   Y O U R   N I G E R I A N          ║
║                    L A W   B U D D Y  🇳🇬                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

  I sabi Nigerian law well well. Ask me anything — formal or casual,
  English or Pidgin. I dey here to help you understand your rights.

  I cover:
    📜  Constitution of the Federal Republic of Nigeria (1999)
    👷  Labour Act — employee rights and workplace law
    🏢  Companies Income Tax Act (CITA)
    💰  Personal Income Tax Act (PITA)
    ⚖️   Criminal Code Act — crimes and penalties
    🗳️   Electoral Act 2022 — election law

  Commands during chat:
    'clear'   — start a fresh conversation
    'quit'    — exit 9jaLawya

  Disclaimer: 9jaLawya provides legal information, not legal advice.
  For serious legal matters, always consult a qualified lawyer.
"""
    print(banner)


# -----------------------------------------------------------------------------
# MAIN CHAT LOOP
# -----------------------------------------------------------------------------

def run_chat(stream: bool = True, web_enabled: bool = True):
    """
    Initialises 9jaLawya and runs the main conversation loop.

    This function:
    1. Imports and initialises NijaLawyaAgent (loads TinyLlama + retriever)
    2. Runs the interactive chat loop
    3. Handles graceful shutdown on Ctrl+C or 'quit'

    Args:
        stream: Whether to stream response tokens as generated
        web_enabled: Whether to allow web search for support resources
    """

    # Import here (not at top of file) so argument parsing and validation
    # run before the heavy model loading begins. This way --help and
    # environment errors are instant, not delayed by 60s of model loading.
    from agent.law_buddy import NijaLawyaAgent

    # Patch web search if disabled by --no-web flag.
    if not web_enabled:
        logging.getLogger(__name__).info(
            "Web search disabled (--no-web flag). "
            "Running in offline mode."
        )
        # Monkey-patch the search tool to return empty results.
        # This avoids touching law_buddy.py's logic.
        import agent.web_search as ws
        ws._search_tool_instance = _DisabledSearchTool()
        # _DisabledSearchTool (defined below) returns empty responses.

    # Initialise the agent — this loads TinyLlama (~60s on first run).
    print("Loading 9jaLawya... (first run downloads TinyLlama ~2.2GB)\n")
    try:
        agent = NijaLawyaAgent()
    except FileNotFoundError as e:
        print(f"\n❌ Setup error: {e}")
        print("Run the ingestion pipeline first. See README for instructions.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to start 9jaLawya: {e}")
        sys.exit(1)

    print_banner()
    print("9jaLawya don ready! How fit I help you today?\n")

    # --- Main chat loop ---
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # User pressed Ctrl+C or pipe closed.
            print("\n\n9jaLawya: E don do! Stay safe and know your rights! 👋\n")
            break

        if not user_input:
            # Empty input — prompt again without any response.
            continue

        if user_input.lower() == "quit":
            print("\n9jaLawya: E don do! Stay safe and know your rights! 👋\n")
            break

        if user_input.lower() == "clear":
            agent.clear_memory()
            print("\n9jaLawya: Memory cleared! We dey start fresh. 😊\n")
            continue

        # Generate response. Streaming prints tokens live to the terminal.
        # Non-streaming waits for full generation then prints at once.
        if not stream:
            # Non-streaming: generate silently, print full response.
            response = agent.chat(user_input, stream=False)
            print(f"\n9jaLawya: {response}\n")
        else:
            # Streaming: law_buddy.py prints "9jaLawya: " prefix and
            # streams tokens directly. We just call chat() and it handles
            # all printing internally via TextStreamer.
            agent.chat(user_input, stream=True)
            print()
            # Extra newline after streamed response for readability.


# -----------------------------------------------------------------------------
# DISABLED SEARCH TOOL (for --no-web mode)
# -----------------------------------------------------------------------------

class _DisabledSearchTool:
    """
    A drop-in replacement for WebSearchTool that always returns empty results.

    Used when --no-web flag is passed. By replacing the singleton with this
    object, we disable web search without changing any logic in law_buddy.py.
    This pattern is called "monkey patching" — swapping an object at runtime.
    """

    def search_for_resources(self, *args, **kwargs):
        from agent.web_search import WebSearchResponse
        return WebSearchResponse(
            results=[], query="", success=False,
            elapsed_seconds=0.0,
            error_message="Web search disabled by --no-web flag."
        )

    def search(self, *args, **kwargs):
        from agent.web_search import WebSearchResponse
        return WebSearchResponse(
            results=[], query="", success=False,
            elapsed_seconds=0.0,
            error_message="Web search disabled by --no-web flag."
        )

    def search_for_legal_aid(self, *args, **kwargs):
        from agent.web_search import WebSearchResponse
        return WebSearchResponse(
            results=[], query="", success=False,
            elapsed_seconds=0.0,
            error_message="Web search disabled by --no-web flag."
        )


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    args = parse_args()
    setup_logging(debug=args.debug)

    # Run environment checks before loading any heavy models.
    env_ok = validate_environment()

    if not env_ok:
        print(
            "❌ Environment check failed. Fix the issues above and try again.\n"
            "   If you haven't run the ingestion pipeline yet, follow these steps:\n\n"
            "   1. Place your PDF files in data/raw/\n"
            "   2. python src/ingestion/pdf_extractor.py\n"
            "   3. python src/ingestion/cleaner.py\n"
            "   4. python src/ingestion/chunker.py\n"
            "   5. python src/ingestion/ingestor.py\n"
            "   6. python main.py\n"
        )
        sys.exit(1)

    # All checks passed — start the chat.
    run_chat(
        stream=not args.no_stream,
        # stream=True by default, False if --no-stream flag was passed.
        web_enabled=not args.no_web,
        # web_enabled=True by default, False if --no-web flag was passed.
    )
