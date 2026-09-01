# src/agent/web_search.py

# Imports
import time
from dataclasses import dataclass
from typing import List, Optional
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# CONSTANTS
MAX_SEARCH_RESULTS = 4
SEARCH_TIMEOUT = 10
RESOURCE_QUERIES = {
    "poverty": "Nigerian government poverty alleviation programmes NSIP NPOWER 2024",
    "unemployment": "Nigeria unemployment benefit job support GEEP NPOWER programme",
    "homeless": "Nigeria homeless shelter welfare support government NGO",
    "food": "Nigeria food support hunger relief government programme 2024",
    "domestic_violence": "Nigeria domestic violence shelter VAPP hotline support",
    "legal_aid": "Nigeria legal aid council free legal services LAC",
    "arrest": "Nigeria legal aid arrest rights police NHRC",
    "discrimination": "Nigeria human rights commission NHRC complaint discrimination",
    "labour": "Nigeria labour ministry complaint unfair dismissal NLC",
    "general_support": "Nigeria social welfare support government programmes 2024",
}

# SEARCH RESULT DATA CLASS
@dataclass
class WebSearchResult:
    """
    Represents a single result from a web search.

    Each result has a title, URL, and snippet (short description).
    The agent uses snippets to build context and URLs to cite sources.
    """

    title: str
    url: str
    snippet: str
    source: str = ""

    def to_context_string(self) -> str:
        """
        Formats this result for LLM prompt injection.

        Returns:
            Formatted context string
        """
        return (
            f"[WEB SOURCE: {self.title}]\n"
            f"[URL: {self.url}]\n"
            f"{self.snippet}"
        )

    def to_display_string(self) -> str:
        """
        Formats this result for display to the user.
        """
        return f"• {self.title}\n  {self.url}"


@dataclass
class WebSearchResponse:
    """
    The complete response from a web search operation.

    Contains all results found, the query used, timing information,
    and a flag indicating whether the search succeeded.
    """

    results: List[WebSearchResult]
    query: str
    success: bool
    elapsed_seconds: float
    error_message: str = ""

    def to_context_string(self) -> str:
        """
        Formats all results as one context block for the LLM.
        """
        if not self.results:
            return "No relevant support resources found online."
        blocks = [r.to_context_string() for r in self.results]
        return "\n\n".join(blocks)

    def to_sources_list(self) -> str:
        """
        Returns a formatted list of sources for display to the user.
        """
        if not self.results:
            return ""
        lines = ["Here are some resources that may help:"]
        for r in self.results:
            lines.append(r.to_display_string())
        return "\n".join(lines)

# WEB SEARCH ENGINE
class WebSearchTool:
    """
    Fetches live web results using DuckDuckGo search.

    Designed to be called sparingly — only when the vector store
    cannot provide the needed information (social support resources,
    legal aid organisations, government programme details).

    Usage:
        tool = WebSearchTool()
        response = tool.search("Nigeria poverty support programmes")
        print(response.to_context_string())
    """

    def __init__(self):
        """
        Initialises the web search tool and verifies the library is available.
        """
        try:
            from ddgs import DDGS
            # DDGS — DuckDuckGo Search class.
            # We import inside __init__ so that if the library is not
            # installed, the error is clear and localised to this class,
            # not a confusing import error at module load time.
            self._ddgs_class = DDGS
            self._available = True
            logger.info("WebSearchTool initialised — DuckDuckGo search available.")
        except ImportError:
            self._available = False
            logger.warning(
                "WebSearchTool: duckduckgo-search not installed. "
                "Web search will be disabled. "
                "Install with: pip install duckduckgo-search"
            )

    def search(
        self,
        query: str,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> WebSearchResponse:
        """
        Performs a web search and returns structured results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            WebSearchResponse with results, success flag, and timing
        """

        if not self._available:
            return WebSearchResponse(
                results=[],
                query=query,
                success=False,
                elapsed_seconds=0.0,
                error_message=(
                    "Web search is not available. "
                    "Install duckduckgo-search: pip install duckduckgo-search"
                ),
            )

        if not query or not query.strip():
            return WebSearchResponse(
                results=[], query=query, success=False,
                elapsed_seconds=0.0, error_message="Empty search query."
            )

        logger.info(f"Web search: '{query[:60]}'")
        start_time = time.time()

        try:
            results = []

            with self._ddgs_class() as ddgs:
                # DDGS() creates a search session.
                # We use it as a context manager (with statement) so the
                # session is properly closed after we're done — prevents
                # connection leaks on repeated searches.

                raw_results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    # max_results limits how many results DuckDuckGo returns.
                    # We already cap at MAX_SEARCH_RESULTS but pass it here
                    # too so DuckDuckGo doesn't fetch more than needed.
                ))
                # ddgs.text() is a generator — list() materialises all results.
                # It returns dicts with keys: title, href, body

            for item in raw_results:
                url = item.get("href", "")
                title = item.get("title", "")
                snippet = item.get("body", "")

                if not snippet or not title:
                    continue
                    # Skip results with no content — they won't help the LLM.

                # Extract domain from URL for clean source attribution.
                source = self._extract_domain(url)

                results.append(WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet[:450],
                    # [:450] caps snippet length — we don't need the full
                    # page description, just enough for the LLM to understand
                    # what the resource offers.
                    source=source,
                ))

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"  Web search complete: {len(results)} results in {elapsed}s")

            return WebSearchResponse(
                results=results,
                query=query,
                success=True,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Web search failed: {e}")
            return WebSearchResponse(
                results=[],
                query=query,
                success=False,
                elapsed_seconds=elapsed,
                error_message=str(e),
            )

    def search_for_resources(self, distress_type: str) -> WebSearchResponse:
        """
        Searches for Nigerian support resources using a pre-crafted query.

        This is the primary method called by law_buddy.py for distress
        scenarios. Using pre-crafted queries (from RESOURCE_QUERIES) is
        more reliable than generating queries from user messages because:
        - Pre-crafted queries are tested and known to return good results
        - User messages may be vague or Pidgin — hard to search directly
        - We can target specific Nigerian government websites

        Args:
            distress_type: Key from RESOURCE_QUERIES dict
                           e.g. "poverty", "domestic_violence", "legal_aid"

        Returns:
            WebSearchResponse with Nigerian support resources
        """

        query = RESOURCE_QUERIES.get(distress_type, RESOURCE_QUERIES["general_support"])
        # .get() with a default ensures we always have a valid query even
        # if an unrecognised distress_type is passed.

        return self.search(query)

    def search_for_legal_aid(self, legal_issue: str) -> WebSearchResponse:
        """
        Searches specifically for legal aid and representation resources.

        Called when a user is in a rights situation and may need a lawyer
        but cannot afford one — a very common scenario in Nigeria.

        Args:
            legal_issue: Brief description of the legal issue

        Returns:
            WebSearchResponse with legal aid resources
        """

        query = f"Nigeria legal aid free lawyer {legal_issue} LAC NHRC"
        return self.search(query)

    def _extract_domain(self, url: str) -> str:
        """
        Extracts the domain name from a URL for clean source citation.

        e.g. "https://nsip.gov.ng/about-us" → "nsip.gov.ng"

        Args:
            url: Full URL string

        Returns:
            Domain string, or empty string if extraction fails
        """
        try:
            # Split on "//" to remove protocol, then take first segment,
            # then split on "/" to remove path.
            return url.split("//")[-1].split("/")[0]
        except Exception:
            return ""

# MODULE-LEVEL SINGLETON
_search_tool_instance: Optional[WebSearchTool] = None

def get_search_tool() -> WebSearchTool:
    """
    Returns the shared WebSearchTool instance (singleton pattern).

    law_buddy.py calls this function rather than instantiating
    WebSearchTool directly — ensures only one instance exists.
    """
    global _search_tool_instance
    if _search_tool_instance is None:
        _search_tool_instance = WebSearchTool()
    return _search_tool_instance


def search_resources(distress_type: str) -> WebSearchResponse:
    """
    Convenience function for resource searching.

    Args:
        distress_type: Key from RESOURCE_QUERIES

    Returns:
        WebSearchResponse
    """
    return get_search_tool().search_for_resources(distress_type)


def search_web(query: str) -> WebSearchResponse:
    """
    Convenience function for general web search.

    Args:
        query: Search query string

    Returns:
        WebSearchResponse
    """
    return get_search_tool().search(query)


# -----------------------------------------------------------------------------
# ENTRY POINT — for testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n9jaLawya Web Search Tool — Test Mode")
    print("="*50)

    tool = WebSearchTool()

    test_queries = [
        ("poverty", "Poverty support resources"),
        ("domestic_violence", "Domestic violence resources"),
        ("legal_aid", "Legal aid resources"),
        ("unemployment", "Unemployment support"),
    ]

    for distress_type, label in test_queries:
        print(f"\n{label}:")
        print("-" * 40)
        response = tool.search_for_resources(distress_type)
        if response.success:
            for r in response.results:
                print(f"  Title:   {r.title}")
                print(f"  URL:     {r.url}")
                print(f"  Snippet: {r.snippet[:100]}...")
                print()
        else:
            print(f"  Search failed: {response.error_message}")
