"""GAIA experiment prompt helpers."""

import os

GAIA_TOOL_DESCRIPTIONS = """
=== AVAILABLE TOOLS ===

1. Web Search (google_web_search):
   - Search the internet for factual information, recent events, or general knowledge
   - Parameter: search_query (string)

2. Calculator (calculator):
   - Evaluate arithmetic expressions: +, -, *, /, **, sqrt, log, sin, cos, etc.
   - Parameter: expression (string)

3. File Reader (file_reader):
   - Read text content from local files: PDF, Excel (.xlsx/.xls), CSV, Word (.docx),
     PowerPoint (.pptx), or plain text (.txt, .py, .json, etc.)
   - Parameters: file_path (string), max_chars (integer, optional, default 20000)

4. Python Executor (python_executor):
   - Execute Python code and return stdout. Use for calculations, data processing,
     or file analysis. Print results to stdout.
   - Parameters: code (string), timeout (integer, optional, default 30s)

5. Multimodal Viewer (multimodal_viewer):
   - Analyse images, audio, or video files using a multimodal AI model.
   - Supported: .png/.jpg/.gif/.webp (images), .mp3/.wav/.m4a (audio), .mp4/.mov (video)
   - Parameters: file_path (string, local path or public URL), prompt (string)

=======================
"""

_MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
}


def get_tool_descriptions() -> str:
    return GAIA_TOOL_DESCRIPTIONS


def enhance_question_with_tools_context(
    question: str, sample: dict, local_file_path: str = ""
) -> str:
    """Append tool descriptions and file hints to the question."""
    parts = [question.strip(), "", get_tool_descriptions()]

    file_name = sample.get("sample_info", {}).get("file_name", "")
    if file_name and local_file_path:
        ext = os.path.splitext(file_name)[1].lower()
        tool_hint = (
            "Use the multimodal_viewer tool to analyse it."
            if ext in _MEDIA_EXTS
            else "Use the file_reader tool to read it."
        )
        parts += [f"Attached File: {file_name}", f"Local Path: {local_file_path}", tool_hint, ""]
    elif file_name:
        parts += [
            f"Attached File: {file_name}",
            "(File not available locally — try web search or python_executor to obtain it.)",
            "",
        ]

    return "\n".join(parts)
