import asyncio
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .constants import MAX_END_DATE, FINAGENT_TASK_TYPE
from .retry import ASYNC_TOOLS_AVAILABLE, backoff, retry_on_429, retry_on_retriable
from .tool_logger import get_tool_call_logger
from .sec_cache import SECQueryCache

try:
    import aiohttp
    from bs4 import BeautifulSoup
except ImportError:
    aiohttp = None
    BeautifulSoup = None

logger = logging.getLogger(__name__)


class ToolDefinition:
    def __init__(self, name: str, body: Dict[str, Any]):
        self.name = name
        self.body = body


class FinancialTool(ABC):
    name: str
    description: str
    input_arguments: Dict[str, Any] = {}
    required_arguments: List[str] = []

    def get_tool_definition(self) -> ToolDefinition:
        body = {
            "name": self.name,
            "description": self.description,
            "properties": self.input_arguments,
            "required": self.required_arguments,
        }
        return ToolDefinition(name=self.name, body=body)

    @abstractmethod
    async def call_tool(self, arguments: dict, **kwargs) -> Dict[str, Any]:
        pass

    async def __call__(
        self,
        arguments: dict = None,
        sample_id: Optional[str] = None,
        batch_num: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        logger.info(f"[TOOL: {self.name.upper()}] Calling with arguments: {arguments}")
        tool_logger = get_tool_call_logger()
        start_time = time.time()
        try:
            tool_result = await self.call_tool(arguments, **kwargs)
            end_time = time.time()
            logger.info(f"[TOOL: {self.name.upper()}] Completed successfully")
            if self.name == "retrieve_information" and isinstance(tool_result, dict):
                result = {
                    "success": True,
                    "result": tool_result.get("retrieval", str(tool_result)),
                    "usage": tool_result.get("usage", {}),
                }
            else:
                result = {
                    "success": True,
                    "result": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result,
                }
            tool_logger.log_call(
                tool_name=self.name, arguments=arguments, result=result,
                start_time=start_time, end_time=end_time, success=True,
                sample_id=sample_id, batch_num=batch_num,
            )
            return result
        except Exception as e:
            end_time = time.time()
            is_verbose = os.environ.get("FINAGENT_VERBOSE", "0") == "1"
            error_msg = str(e)
            if is_verbose:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
                logger.warning(f"[TOOL: {self.name.upper()}] Error: {e}\nTraceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[TOOL: {self.name.upper()}] Error: {error_msg}")
            result = {"success": False, "result": error_msg}
            tool_logger.log_call(
                tool_name=self.name, arguments=arguments, result=result,
                start_time=start_time, end_time=end_time, success=False,
                error_message=error_msg, sample_id=sample_id, batch_num=batch_num,
            )
            return result


class GoogleWebSearch(FinancialTool):
    name: str = "google_web_search"
    description: str = "Search the web for information"
    input_arguments: Dict[str, Any] = {
        "search_query": {"type": "string", "description": "The query to search for"}
    }
    required_arguments: List[str] = ["search_query"]

    def __init__(
        self,
        top_n_results: int = 5,
        serpapi_api_key: str = None,
        serper_api_key: str = None,
    ):
        self.top_n_results = top_n_results
        self.serpapi_api_key = serpapi_api_key or os.getenv("SERPAPI_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self._api_provider = self._determine_api_provider()

    def _determine_api_provider(self) -> str:
        if self.serpapi_api_key:
            return "serpapi"
        elif self.serper_api_key:
            return "serper"
        return "none"

    async def _execute_search_serpapi(self, search_query: str) -> List[Dict]:
        if not self.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is not set")
        max_date_parts = MAX_END_DATE.split("-")
        google_date_format = f"{max_date_parts[1]}/{max_date_parts[2]}/{max_date_parts[0]}"
        params = {
            "api_key": self.serpapi_api_key,
            "engine": "google",
            "q": search_query,
            "num": self.top_n_results,
            "tbs": f"cdr:1,cd_max:{google_date_format}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get("https://serpapi.com/search.json", params=params) as response:
                response.raise_for_status()
                results = await response.json()
        logger.debug(f"SerpAPI returned {len(results.get('organic_results', []))} results")
        return results.get("organic_results", [])

    async def _execute_search_serper(self, search_query: str) -> List[Dict]:
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY is not set")
        headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
        payload = {"q": search_query, "num": self.top_n_results}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://google.serper.dev/search", headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                results = await response.json()
        organic_results = results.get("organic", [])
        normalized_results = []
        for item in organic_results[: self.top_n_results]:
            normalized_results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", 0),
            })
        logger.debug(f"Serper API returned {len(normalized_results)} results")
        return normalized_results

    async def _execute_search(self, search_query: str) -> List[Dict]:
        if self.serpapi_api_key:
            try:
                logger.info("[GOOGLE_WEB_SEARCH] Using SerpAPI for search")
                return await self._execute_search_serpapi(search_query)
            except Exception as e:
                logger.warning(f"SerpAPI failed: {e}. Attempting fallback to Serper API...")
        if self.serper_api_key:
            try:
                logger.info("[GOOGLE_WEB_SEARCH] Using Serper API for search")
                return await self._execute_search_serper(search_query)
            except Exception as e:
                logger.warning(f"Serper API also failed: {e}")
                raise ValueError(f"Both SerpAPI and Serper API failed. Last error: {e}")
        raise ValueError("No search API key available. Please set SERPAPI_API_KEY or SERPER_API_KEY")

    async def call_tool(self, arguments: dict, **kwargs) -> List[Dict]:
        search_query = arguments.get("search_query") or arguments.get("query", "")
        if not ASYNC_TOOLS_AVAILABLE:
            logger.warning("aiohttp not available, returning mock results")
            return self._mock_search_results(search_query)
        if self._api_provider == "none":
            logger.warning("No search API key configured, returning mock results")
            return self._mock_search_results(search_query)
        if backoff is not None:
            return await retry_on_429(self._execute_search)(search_query)
        return await self._execute_search(search_query)

    def _mock_search_results(self, query: str) -> List[Dict]:
        return [{"title": f"Search result for: {query}", "link": "https://example.com", "snippet": f"Mock search result for query: {query}"}]

    def get_api_status(self) -> Dict[str, Any]:
        return {
            "active_provider": self._api_provider,
            "serpapi_configured": bool(self.serpapi_api_key),
            "serper_configured": bool(self.serper_api_key),
        }


class EDGARSearch(FinancialTool):
    name: str = "edgar_search"
    description: str = """
    Search the EDGAR Database through the SEC API.
    You should provide a query, a list of form types, a list of CIKs, a start date, an end date, a page number, and a top N results.
    The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.
    """.strip()
    input_arguments: Dict[str, Any] = {
        "query": {"type": "string", "description": "The keyword or phrase to search, such as 'substantial doubt' OR 'material weakness'"},
        "form_types": {"type": "array", "description": "Limits search to specific SEC form types (e.g., ['8-K', '10-Q']) list of strings. Default is None (all form types)", "items": {"type": "string"}},
        "ciks": {"type": "array", "description": "Filters results to filings by specified CIKs, type list of strings. Default is None (all filers).", "items": {"type": "string"}},
        "start_date": {"type": "string", "description": "Start date for the search range in yyyy-mm-dd format. Used with endDate to define the date range. Example: '2024-01-01'. Default is 30 days ago"},
        "end_date": {"type": "string", "description": "End date for the search range, in the same format as startDate. Default is today"},
        "page": {"type": "string", "description": "Pagination for results. Default is '1'"},
        "top_n_results": {"type": "integer", "description": "The top N results to return after the query. Useful if you are not sure the result you are looking for is ranked first after your query."},
    }
    required_arguments: List[str] = ["query", "form_types", "ciks", "start_date", "end_date", "page", "top_n_results"]

    def __init__(self, sec_api_key: str = None, cache_dir: str = None):
        self.sec_api_key = sec_api_key or os.getenv("SEC_EDGAR_API_KEY")
        self.sec_api_url = "https://api.sec-api.io/full-text-search"
        self.cache = SECQueryCache(cache_dir=cache_dir)

    async def _execute_search(self, query, form_types, ciks, start_date, end_date, page, top_n_results) -> List[Dict]:
        if not self.sec_api_key:
            raise ValueError("SEC_EDGAR_API_KEY is not set")
        if isinstance(form_types, str) and form_types.startswith("[") and form_types.endswith("]"):
            try:
                form_types = json.loads(form_types.replace("'", '"'))
            except json.JSONDecodeError:
                form_types = [item.strip(' "\"') for item in form_types[1:-1].split(",")]
        if isinstance(ciks, str) and ciks.startswith("[") and ciks.endswith("]"):
            try:
                ciks = json.loads(ciks.replace("'", '"'))
            except json.JSONDecodeError:
                ciks = [item.strip(' "\"') for item in ciks[1:-1].split(",")]
        if end_date and end_date > MAX_END_DATE:
            end_date = MAX_END_DATE
        payload = {"query": query, "formTypes": form_types, "ciks": ciks, "startDate": start_date, "endDate": end_date, "page": page}
        headers = {"Content-Type": "application/json", "Authorization": self.sec_api_key}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.sec_api_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
        return result.get("filings", [])[:int(top_n_results)]

    async def call_tool(self, arguments: dict, **kwargs) -> List[Dict]:
        query = arguments.get("query", "")
        form_types = arguments.get("form_types", [])
        ciks = arguments.get("ciks", [])
        start_date = arguments.get("start_date", "")
        end_date = arguments.get("end_date", "")
        page = arguments.get("page", "1")
        top_n_results = arguments.get("top_n_results", 5)
        if not ASYNC_TOOLS_AVAILABLE:
            return self._mock_edgar_results(query)
        cached_result = self.cache.get(query, form_types, ciks, start_date, end_date, page, top_n_results)
        if cached_result is not None:
            return cached_result
        try:
            if backoff is not None:
                result = await retry_on_429(self._execute_search)(query, form_types, ciks, start_date, end_date, page, top_n_results)
            else:
                result = await self._execute_search(query, form_types, ciks, start_date, end_date, page, top_n_results)
            self.cache.put(query, form_types, ciks, start_date, end_date, page, top_n_results, result)
            return result
        except Exception as e:
            is_verbose = os.environ.get("FINAGENT_VERBOSE", "0") == "1"
            if is_verbose:
                logger.error(f"SEC API error: {e}\nTraceback: {traceback.format_exc()}")
            else:
                logger.error(f"SEC API error: {e}")
            raise

    def _mock_edgar_results(self, query: str) -> List[Dict]:
        return [{"cik": "0001234567", "company_name": "Example Corp", "form_type": "10-K", "filing_date": "2024-03-15", "query_match": query}]


class ParseHtmlPage(FinancialTool):
    name: str = "parse_html_page"
    description: str = """
    Parse an HTML page. This tool is used to parse the HTML content of a page and saves the content outside of the conversation to avoid context window issues.
    You should provide both the URL of the page to parse, as well as the key you want to use to save the result in the agent's data structure.
    The data structure is a dictionary.
    """.strip()
    input_arguments: Dict[str, Any] = {
        "url": {"type": "string", "description": "The URL of the HTML page to parse"},
        "key": {"type": "string", "description": "The key to use when saving the result in the conversation's data structure (dict)."},
    }
    required_arguments: List[str] = ["url", "key"]

    def __init__(self, headers: dict = None):
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    async def _parse_html_page(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers, timeout=60) as response:
                    response.raise_for_status()
                    html_content = await response.text()
            except Exception as e:
                if len(str(e)) == 0:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds. "
                        "The URL might be blocked or the server is taking too long to respond."
                    )
                else:
                    is_verbose = os.environ.get("FINAGENT_VERBOSE", "0") == "1"
                    if is_verbose:
                        raise Exception(str(e) + "\nTraceback: " + traceback.format_exc())
                    else:
                        raise Exception(str(e))
        soup = BeautifulSoup(html_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        return text

    async def _save_tool_output(self, output: str, key: str, data_storage: dict) -> str:
        if not output:
            return "ERROR: No content to save"
        tool_result = ""
        if key in data_storage:
            tool_result = "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
        tool_result += f"SUCCESS: The result has been saved to the data storage under the key: {key}.\n"
        data_storage[key] = output
        keys_list = "\n".join(data_storage.keys())
        tool_result += f"The data_storage currently contains the following keys:\n{keys_list}\n"
        return tool_result

    async def call_tool(self, arguments: dict, data_storage: dict = None, **kwargs) -> str:
        url = arguments.get("url", "")
        key = arguments.get("key", "parsed_content")
        if data_storage is None:
            data_storage = {}
        if not ASYNC_TOOLS_AVAILABLE:
            content = f"Mock content from {url}"
            data_storage[key] = content
            return f"SUCCESS: Content saved under key '{key}'"
        if backoff is not None:
            text_output = await retry_on_retriable(self._parse_html_page)(url)
        else:
            text_output = await self._parse_html_page(url)
        return await self._save_tool_output(text_output, key, data_storage)


class RetrieveInformation(FinancialTool):
    import re as _re
    name: str = "retrieve_information"
    description: str = """
    Retrieve information from the conversation's data structure (dict) and allow character range extraction.

    IMPORTANT: Your prompt MUST include at least one key from the data storage using the exact format: {{key_name}}

    For example, if you want to analyze data stored under the key "financial_report", your prompt should look like:
    "Analyze the following financial report and extract the revenue figures: {{financial_report}}"

    The {{key_name}} will be replaced with the actual content stored under that key before being sent to the LLM.
    If you don't use this exact format with double braces, the tool will fail to retrieve the information.

    You can optionally specify character ranges for each document key to extract only portions of documents. That can be useful to avoid token limit errors or improve efficiency by selecting only part of the document.
    For example, if "financial_report" contains "Annual Report 2023" and you specify a range [1, 5] for that key,
    only "nnual" will be inserted into the prompt.

    The output is the result from the LLM that receives the prompt with the inserted data.
    """.strip()
    input_arguments: Dict[str, Any] = {
        "prompt": {"type": "string", "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder."},
        "input_character_ranges": {"type": "object", "description": "A dictionary mapping document keys to their character ranges. Each range should be an array where the first element is the start index and the second element is the end index. Can be used to only read portions of documents. By default, the full document is used. To use the full document, set the range to an empty list [].", "additionalProperties": {"type": "array", "items": {"type": "integer"}}},
    }
    required_arguments: List[str] = ["prompt"]

    async def call_tool(
        self,
        arguments: dict,
        data_storage: dict = None,
        llm_func: Callable = None,
        **kwargs,
    ) -> Dict[str, Any]:
        import re
        prompt = arguments.get("prompt", "")
        input_character_ranges = arguments.get("input_character_ranges", {}) or {}
        if data_storage is None:
            data_storage = {}
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. "
                "Please try again with the correct format."
            )
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        formatted_data = {}
        for key in keys:
            if key not in data_storage:
                raise KeyError(
                    f"ERROR: The key '{key}' was not found in the data storage. "
                    f"Available keys are: {', '.join(data_storage.keys())}"
                )
            doc_content = data_storage[key]
            if key in input_character_ranges:
                char_range = input_character_ranges[key]
                if len(char_range) == 0:
                    formatted_data[key] = doc_content
                elif len(char_range) != 2:
                    raise ValueError(
                        f"ERROR: The character range for key '{key}' must be a list with two elements or an empty list. "
                        "Please try again with the correct format."
                    )
                else:
                    formatted_data[key] = doc_content[int(char_range[0]):int(char_range[1])]
            else:
                formatted_data[key] = doc_content
        formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)
        try:
            final_prompt = formatted_prompt.format(**formatted_data)
        except KeyError as e:
            raise KeyError(
                f"ERROR: The key {str(e)} was not found in the data storage. "
                f"Available keys are: {', '.join(data_storage.keys())}"
            )
        if llm_func:
            if asyncio.iscoroutinefunction(llm_func):
                response = await llm_func(final_prompt)
            else:
                response = llm_func(final_prompt)
            if hasattr(response, "output_text_str"):
                return {"retrieval": response.output_text_str, "usage": getattr(response, "metadata", {})}
            elif isinstance(response, dict):
                return {"retrieval": response.get("output", str(response)), "usage": response.get("usage", {})}
            else:
                return {"retrieval": str(response), "usage": {}}
        return {"retrieval": f"Formatted prompt (no LLM provided):\n{final_prompt[:1000]}...", "usage": {}}


def create_financial_tools(
    serpapi_api_key: str = None,
    serper_api_key: str = None,
    sec_api_key: str = None,
) -> Dict[str, FinancialTool]:
    return {
        "google_web_search": GoogleWebSearch(serpapi_api_key=serpapi_api_key, serper_api_key=serper_api_key),
        "edgar_search": EDGARSearch(sec_api_key=sec_api_key),
        "parse_html_page": ParseHtmlPage(),
        "retrieve_information": RetrieveInformation(),
    }


def get_all_tool_definitions(tools: Dict[str, FinancialTool]) -> List[Dict[str, Any]]:
    definitions = []
    for tool in tools.values():
        tool_def = tool.get_tool_definition()
        definitions.append({
            "type": "function",
            "function": {
                "name": tool_def.body["name"],
                "description": tool_def.body["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_def.body["properties"],
                    "required": tool_def.body["required"],
                },
            },
        })
    return definitions
