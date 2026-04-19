import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolCallLogger:
    def __init__(self):
        self._call_records: List[Dict[str, Any]] = []
        self._session_start = datetime.now()

    def log_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        start_time: float,
        end_time: float,
        success: bool,
        error_message: Optional[str] = None,
        sample_id: Optional[str] = None,
        batch_num: Optional[int] = None,
    ) -> None:
        record = {
            "call_id": len(self._call_records) + 1,
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": self._sanitize_arguments(arguments),
            "result_summary": self._summarize_result(result),
            "execution_time_ms": round((end_time - start_time) * 1000, 2),
            "success": success,
            "error_message": error_message,
            "sample_id": sample_id,
            "batch_num": batch_num,
        }
        self._call_records.append(record)

    def _sanitize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if arguments is None:
            return {}
        sanitized = {}
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > 500:
                sanitized[key] = value[:500] + f"... (truncated, total {len(value)} chars)"
            else:
                sanitized[key] = value
        return sanitized

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result is None:
            return {"type": "none"}
        summary = {
            "success": result.get("success", False),
            "result_type": type(result.get("result", None)).__name__,
        }
        result_content = result.get("result", "")
        if isinstance(result_content, str):
            summary["result_length"] = len(result_content)
            if len(result_content) <= 1000:
                summary["result_preview"] = result_content
            else:
                summary["result_preview"] = result_content[:1000] + f"... (truncated, total {len(result_content)} chars)"
        elif isinstance(result_content, (list, dict)):
            try:
                json_str = json.dumps(result_content)
                summary["result_length"] = len(json_str)
                if len(json_str) <= 1000:
                    summary["result_preview"] = result_content
                else:
                    summary["result_preview"] = json_str[:1000] + "... (truncated)"
            except (TypeError, ValueError):
                summary["result_preview"] = str(result_content)[:500]
        return summary

    def get_statistics(self) -> Dict[str, Any]:
        if not self._call_records:
            return {"total_calls": 0}
        tool_stats = defaultdict(lambda: {
            "count": 0, "success_count": 0, "total_time_ms": 0, "errors": [],
        })
        for record in self._call_records:
            name = record["tool_name"]
            tool_stats[name]["count"] += 1
            tool_stats[name]["total_time_ms"] += record["execution_time_ms"]
            if record["success"]:
                tool_stats[name]["success_count"] += 1
            elif record["error_message"]:
                tool_stats[name]["errors"].append(record["error_message"])
        for stats in tool_stats.values():
            if stats["count"] > 0:
                stats["avg_time_ms"] = round(stats["total_time_ms"] / stats["count"], 2)
                stats["success_rate"] = round(stats["success_count"] / stats["count"], 4)
        return {
            "total_calls": len(self._call_records),
            "session_duration_seconds": (datetime.now() - self._session_start).total_seconds(),
            "by_tool": dict(tool_stats),
        }

    def get_all_records(self) -> List[Dict[str, Any]]:
        return self._call_records.copy()

    def export_to_file(self, filepath: str) -> None:
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "session_start": self._session_start.isoformat(),
            "statistics": self.get_statistics(),
            "call_records": self._call_records,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Tool call logs exported to: {filepath}")

    def clear(self) -> None:
        self._call_records.clear()
        self._session_start = datetime.now()


_tool_call_logger: Optional[ToolCallLogger] = None


def get_tool_call_logger() -> ToolCallLogger:
    global _tool_call_logger
    if _tool_call_logger is None:
        _tool_call_logger = ToolCallLogger()
    return _tool_call_logger


def reset_tool_call_logger() -> None:
    global _tool_call_logger
    _tool_call_logger = ToolCallLogger()
