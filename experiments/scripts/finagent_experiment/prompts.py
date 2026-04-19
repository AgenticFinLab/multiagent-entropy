from .constants import MAX_END_DATE, FINAGENT_TASK_TYPE

from maep.prompts import (
    TASK_IDENTIFIERS,
    get_identifier,
    validate_task_type,
)


FINAGENT_INSTRUCTIONS_PROMPT = """
You are an expert financial analyst AI designed to answer complex financial questions.
Your task is to provide accurate, well-researched answers using the available tools.

Available Tools:
1. google_web_search: Search the web for financial information, news, and analysis
   - Required: search_query (string)

2. edgar_search: Search SEC EDGAR database for company filings (10-K, 10-Q, 8-K, etc.)
   - Required: query, form_types, ciks, start_date, end_date, page, top_n_results
   - Note: Dates must be in yyyy-mm-dd format, end_date cannot exceed {max_end_date}

3. parse_html_page: Parse HTML pages to extract content and save to data storage
   - Required: url, key (the key to save the content under)

4. retrieve_information: Analyze stored information using {{{{key_name}}}} format
   - Required: prompt (must include {{{{key_name}}}} placeholders)
   - Optional: input_character_ranges (dict mapping keys to [start, end] ranges)

IMPORTANT GUIDELINES:
- Use tools strategically to gather accurate financial data
- For SEC filings, use edgar_search with specific queries and form types
- Parse relevant HTML pages and save content for later analysis with retrieve_information
- After at most 5 tool uses, provide your FINAL ANSWER
- Always cite your sources

When ready with your answer, respond with:
FINAL ANSWER: [your detailed answer here]
{{"sources": [list of sources used]}}

Question: {question}
""".replace("{max_end_date}", MAX_END_DATE)


def get_tool_descriptions() -> str:
    descriptions = [
        "\n=== AVAILABLE FINANCIAL TOOLS ===",
        "",
        "1. Web Search (google_web_search):",
        "   - Search the web for financial information, news, and analysis",
        "   - Useful for recent news, market data, and general financial queries",
        "   - Parameter: search_query (string)",
        "",
        "2. SEC EDGAR Search (edgar_search):",
        "   - Search SEC's EDGAR database for official company filings",
        "   - Supports: 10-K (annual), 10-Q (quarterly), 8-K (current events)",
        "   - Filter by CIK, form type, date range",
        "   - Parameters: query, form_types, ciks, start_date, end_date, page, top_n_results",
        f"   - Note: end_date cannot exceed {MAX_END_DATE}",
        "",
        "3. HTML Parser (parse_html_page):",
        "   - Extract text content from web pages and SEC filing URLs",
        "   - Saves content to data storage for later analysis",
        "   - Parameters: url, key (storage key name)",
        "",
        "4. Information Retrieval (retrieve_information):",
        "   - Analyze stored content from parsed pages using LLM",
        "   - Use {{key_name}} format to reference saved data in your prompt",
        "   - Parameters: prompt (with {{key}} placeholders), input_character_ranges (optional)",
        "",
        "==================================",
    ]
    return "\n".join(descriptions)


def enhance_question_with_tools_context(
    question: str,
    sample: dict,
    task_type: str = FINAGENT_TASK_TYPE,
    use_maep_prompts: bool = True,
) -> str:
    question_type = sample.get("sample_info", {}).get("question_type", "")
    task = sample.get("sample_info", {}).get("task", "")

    try:
        answer_identifier = get_identifier(task_type)
    except ValueError:
        answer_identifier = TASK_IDENTIFIERS.get("finance", "FINAL ANSWER:")

    enhanced = ["=== FINANCIAL ANALYSIS TASK ===", ""]

    if task:
        enhanced.append(f"Task Category: {task}")
    if question_type:
        enhanced.append(f"Question Type: {question_type}")
    if task or question_type:
        enhanced.append("")

    enhanced.extend([
        "QUESTION:",
        question,
        "",
        get_tool_descriptions(),
        "",
        f"Note: SEC data searches are limited to dates before {MAX_END_DATE}",
        "",
    ])

    if use_maep_prompts and validate_task_type(task_type):
        enhanced.extend([
            "ANSWER FORMAT REQUIREMENT:",
            f"Mark your final answer clearly using the format: {answer_identifier} [your answer]",
            "",
        ])

    return "\n".join([line for line in enhanced if line or line == ""])
