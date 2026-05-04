"""FlySuiteFOS graph learning hooks for Hermes.

This plugin observes read-only FlySuiteFOS graph MCP tool calls, keeps a compact
local notebook of useful graph-query discoveries, and mirrors safe summaries to
FSA memory. It does not query FOS directly and does not write to FOS.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

FOS_TOOL_PREFIX = "mcp_flysuite_fos_graph_"
FOS_SOURCE_CHANNEL = "hermes_fos_graph"

_FOS_QUERY_RE = re.compile(
    r"(FOS|FHM|OMS|WMS|ERP|POS|CS|圖譜|關係|ontology|graph|entity|entities|"
    r"claim|relation|path|timeline|object.?set|商品|庫存|倉庫|訂單|明細|"
    r"銷貨|銷退|進貨|寄倉|物流|出貨|客戶|客服|工單|廠商|通路)",
    re.IGNORECASE,
)
_SECRET_KEY_RE = re.compile(r"(password|secret|token|api[_-]?key|aes|credential)", re.IGNORECASE)
_PII_KEY_RE = re.compile(
    r"(phone|email|address|tax|recipient|buyer_name|contact_name|line_uuid|"
    r"line_picture_url|raw_payload|api_response|print_data)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s()]{7,}\d)(?!\d)")
_TOKEN_RE = re.compile(r"\b(?:sk|ghp|gho|xox[baprs]|line)[-_A-Za-z0-9]{16,}\b", re.IGNORECASE)
_LOCAL_URL_RE = re.compile(r"http://(?:127\.0\.0\.1|localhost):\d+(?:/[^\s\"']*)?", re.IGNORECASE)


@dataclass
class FosGraphLearningConfig:
    enabled: bool = True
    sync_enabled: bool = True
    fsa_base_url: str = "http://127.0.0.1:18793"
    context_limit: int = 6
    pending_limit: int = 12
    max_tool_result_chars: int = 28000
    local_history_limit: int = 240
    fsa_timeout_seconds: float = 2.0


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cfg_get(key: str, default: Any = None) -> Any:
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("fos_graph_learning") or {}
        return section.get(key, default) if isinstance(section, dict) else default
    except Exception:
        return default


def _load_config() -> FosGraphLearningConfig:
    enabled = _truthy(os.getenv("FOS_GRAPH_LEARNING_ENABLED", _cfg_get("enabled")), True)
    sync_enabled = _truthy(
        os.getenv("FOS_GRAPH_LEARNING_SYNC_ENABLED", _cfg_get("sync_enabled")),
        True,
    )
    fsa_base_url = str(
        os.getenv("FOS_GRAPH_LEARNING_FSA_BASE_URL")
        or _cfg_get("fsa_base_url")
        or "http://127.0.0.1:18793"
    ).rstrip("/")
    context_limit = _int_value(
        os.getenv("FOS_GRAPH_LEARNING_CONTEXT_LIMIT", _cfg_get("context_limit")),
        6,
    )
    pending_limit = _int_value(
        os.getenv("FOS_GRAPH_LEARNING_PENDING_LIMIT", _cfg_get("pending_limit")),
        12,
    )
    max_tool_result_chars = _int_value(
        os.getenv("FOS_GRAPH_LEARNING_MAX_TOOL_RESULT_CHARS", _cfg_get("max_tool_result_chars")),
        28000,
    )
    local_history_limit = _int_value(
        os.getenv("FOS_GRAPH_LEARNING_LOCAL_HISTORY_LIMIT", _cfg_get("local_history_limit")),
        240,
    )
    fsa_timeout_seconds = _float_value(
        os.getenv("FOS_GRAPH_LEARNING_FSA_TIMEOUT_SECONDS", _cfg_get("fsa_timeout_seconds")),
        2.0,
    )
    return FosGraphLearningConfig(
        enabled=enabled,
        sync_enabled=sync_enabled,
        fsa_base_url=fsa_base_url,
        context_limit=max(1, min(20, context_limit)),
        pending_limit=max(1, min(50, pending_limit)),
        max_tool_result_chars=max(4000, max_tool_result_chars),
        local_history_limit=max(20, min(2000, local_history_limit)),
        fsa_timeout_seconds=max(0.2, fsa_timeout_seconds),
    )


def _is_fos_tool(tool_name: str) -> bool:
    return tool_name.startswith(FOS_TOOL_PREFIX) and tool_name[len(FOS_TOOL_PREFIX):].startswith("fos_")


def _is_fos_query(text: str) -> bool:
    return bool(text and _FOS_QUERY_RE.search(text))


def _sanitize_text(value: Any, *, max_chars: int = 1200) -> str:
    text = str(value or "").replace("\x00", " ")
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _TOKEN_RE.sub("[secret]", text)
    text = _LOCAL_URL_RE.sub("本機服務", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    return text


def _sanitize_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _SECRET_KEY_RE.search(key):
        return "[REDACTED_SECRET]"
    if key == "databaseUrl":
        return "[REDACTED_DATABASE_URL]"
    if _PII_KEY_RE.search(key):
        return "[REDACTED_PII]"
    if depth > 5:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            str(k): _sanitize_value(v, key=str(k), depth=depth + 1)
            for k, v in value.items()
            if str(k) not in {"raw_payload", "api_response", "print_data"}
        }
    if isinstance(value, list):
        return [_sanitize_value(item, key=key, depth=depth + 1) for item in value[:40]]
    if isinstance(value, str):
        return _sanitize_text(value, max_chars=1200)
    return value


def _parse_tool_payload(result: Any) -> Any:
    if isinstance(result, str):
        try:
            outer = json.loads(result)
        except json.JSONDecodeError:
            return result
    else:
        outer = result

    if isinstance(outer, dict) and "result" in outer:
        payload = outer.get("result")
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return payload
        return payload
    return outer


def _compact_list(items: list[Any], *, limit: int = 10) -> dict[str, Any]:
    return {
        "count": len(items),
        "preview": [_compact_payload(item, depth=1) for item in items[:limit]],
        "truncated": len(items) > limit,
    }


def _compact_payload(payload: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[TRUNCATED]"
    if isinstance(payload, list):
        return _compact_list(payload, limit=8)
    if not isinstance(payload, dict):
        return _sanitize_value(payload, depth=depth)

    compact: dict[str, Any] = {}
    for key in (
        "ok",
        "type",
        "mode",
        "seed_id",
        "depth",
        "direction",
        "fromType",
        "toType",
        "from_type",
        "to_type",
        "specialTablePolicy",
    ):
        if key in payload:
            compact[key] = payload[key]

    if isinstance(payload.get("summary"), dict):
        summary = dict(payload["summary"])
        summary.pop("databaseUrl", None)
        compact["summary"] = _sanitize_value(summary)

    for key in ("lineage", "piiPolicy", "current"):
        if key in payload:
            compact[key] = _sanitize_value(payload[key])

    if isinstance(payload.get("rows"), list):
        compact["rows"] = _compact_list(payload["rows"], limit=12)
    if isinstance(payload.get("entityTypes"), list):
        compact["entityTypes"] = _compact_list(
            [_compact_entity_type(item) for item in payload["entityTypes"]],
            limit=20,
        )
    if isinstance(payload.get("readModels"), list):
        compact["readModels"] = _compact_list(payload["readModels"], limit=10)
    if isinstance(payload.get("nodes"), list):
        compact["nodes"] = _compact_list(payload["nodes"], limit=20)
    if isinstance(payload.get("edges"), list):
        compact["edges"] = _compact_list(payload["edges"], limit=30)
    if isinstance(payload.get("paths"), list):
        compact["paths"] = _compact_list(payload["paths"], limit=12)
    if isinstance(payload.get("entity"), dict):
        compact["entity"] = _sanitize_value(payload["entity"])
    if isinstance(payload.get("claims"), dict):
        compact["claims"] = _sanitize_value(payload["claims"])
    if isinstance(payload.get("relations"), list):
        compact["relations"] = _compact_list(payload["relations"], limit=20)
    if isinstance(payload.get("events"), list):
        compact["events"] = _compact_list(payload["events"], limit=20)
    if "error" in payload:
        compact["error"] = _sanitize_text(payload.get("error"), max_chars=800)

    if not compact:
        for key, value in list(payload.items())[:12]:
            compact[key] = _compact_payload(value, depth=depth + 1)
    return compact


def _compact_entity_type(item: Any) -> Any:
    if not isinstance(item, dict):
        return _sanitize_value(item)
    relations = item.get("relationPaths") if isinstance(item.get("relationPaths"), list) else []
    return {
        "type": item.get("type"),
        "label": item.get("label"),
        "domain": item.get("domain"),
        "primaryIdentifier": item.get("primaryIdentifier"),
        "displayClaims": item.get("displayClaims"),
        "searchClaims": item.get("searchClaims"),
        "sourceOfTruth": item.get("sourceOfTruth"),
        "activeCount": item.get("activeCount"),
        "relationsPreview": [
            {
                "subjectType": rel.get("subjectType"),
                "predicate": rel.get("predicate"),
                "objectType": rel.get("objectType"),
                "label": rel.get("label"),
                "count": rel.get("count"),
            }
            for rel in relations[:8]
            if isinstance(rel, dict)
        ],
    }


def _tool_short_name(tool_name: str) -> str:
    return tool_name[len(FOS_TOOL_PREFIX):] if tool_name.startswith(FOS_TOOL_PREFIX) else tool_name


def _tool_action_label(tool_name: str) -> str:
    short = _tool_short_name(tool_name)
    labels = {
        "fos_graph_health": "FOS 圖譜健康檢查",
        "fos_describe_ontology": "FOS ontology 查詢",
        "fos_search_entities": "FOS entity 搜尋",
        "fos_get_entity": "FOS entity 詳查",
        "fos_get_timeline": "FOS timeline 查詢",
        "fos_explain_path": "FOS 關係路徑說明",
        "fos_traverse_graph": "FOS 圖譜關聯查詢",
        "fos_query_object_set": "FOS object-set 查詢",
    }
    return labels.get(short, "FOS 圖譜查詢")


def _summarize_tool_observation(tool_name: str, args: dict[str, Any], compact: Any) -> str:
    action = _tool_action_label(tool_name)
    arg_bits = []
    for key in ("q", "type", "domain", "from_type", "to_type", "seed_id", "id"):
        if args.get(key):
            arg_bits.append(f"{key}={_sanitize_text(args[key], max_chars=80)}")
    if isinstance(args.get("spec"), dict):
        spec = args["spec"]
        from_type = ((spec.get("from") or {}).get("type") if isinstance(spec.get("from"), dict) else None)
        if from_type:
            arg_bits.append(f"from={from_type}")
        if spec.get("aggregate"):
            arg_bits.append("aggregate=true")
    arg_text = ", ".join(arg_bits) if arg_bits else "no-filter"

    if isinstance(compact, dict):
        if isinstance(compact.get("summary"), dict):
            summary = compact["summary"]
            return (
                f"{action}({arg_text}) confirmed registry scale: "
                f"{summary.get('entityTypes')} entity types, "
                f"{summary.get('relationPaths')} relation paths, "
                f"{summary.get('claimPredicates')} claim predicates."
            )
        if isinstance(compact.get("entityTypes"), dict):
            previews = compact["entityTypes"].get("preview") or []
            labels = [
                f"{item.get('type')}({item.get('label')})"
                for item in previews[:5]
                if isinstance(item, dict) and item.get("type")
            ]
            return f"{action}({arg_text}) surfaced entity types: {', '.join(labels)}."
        if isinstance(compact.get("rows"), dict):
            rows = compact["rows"]
            return f"{action}({arg_text}) returned {rows.get('count')} rows in {compact.get('mode', 'query')} mode."
        if isinstance(compact.get("nodes"), dict) or isinstance(compact.get("edges"), dict):
            return (
                f"{action}({arg_text}) traversed graph with "
                f"{(compact.get('nodes') or {}).get('count', 0)} nodes and "
                f"{(compact.get('edges') or {}).get('count', 0)} edges."
            )
        if compact.get("error"):
            return f"{action}({arg_text}) failed: {compact.get('error')}."
    return f"{action}({arg_text}) returned FOS graph context."


class FosGraphLearningState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, list[dict[str, Any]]] = {}

    def record_tool_call(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        session_id: str,
        duration_ms: int = 0,
    ) -> None:
        config = _load_config()
        if not config.enabled or not _is_fos_tool(tool_name):
            return
        payload = _parse_tool_payload(result)
        compact = _compact_payload(payload)
        observation = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": tool_name,
            "args": _sanitize_value(args),
            "duration_ms": duration_ms,
            "compact": compact,
            "note": _summarize_tool_observation(tool_name, args, compact),
        }
        key = session_id or "default"
        with self._lock:
            bucket = self._pending.setdefault(key, [])
            bucket.append(observation)
            if len(bucket) > config.pending_limit:
                del bucket[: len(bucket) - config.pending_limit]

    def pop_pending(self, session_id: str) -> list[dict[str, Any]]:
        key = session_id or "default"
        with self._lock:
            return self._pending.pop(key, [])

    def transform_result(self, *, tool_name: str, result: Any) -> Optional[str]:
        config = _load_config()
        if not config.enabled or not _is_fos_tool(tool_name):
            return None
        if not isinstance(result, str) or len(result) <= config.max_tool_result_chars:
            return None
        payload = _parse_tool_payload(result)
        compact = _compact_payload(payload)
        return json.dumps(
            {
                "result": compact,
                "truncated_by": "fos_graph_learning",
                "note": (
                    "Large FOS graph output was compacted for the model. "
                    "Ask for a narrower entity, relation, timeline, or object-set query if more detail is needed."
                ),
            },
            ensure_ascii=False,
        )

    def build_context(self, user_message: str) -> Optional[dict[str, str]]:
        config = _load_config()
        if not config.enabled or not _is_fos_query(user_message):
            return None
        notes = self._load_recent_notes(user_message, limit=config.context_limit)
        lines = [
            "FOS 圖譜查詢工作方式：",
            "- 先用 ontology/entity type 查清楚資料語意，再查 entity、timeline、graph traversal 或 object-set。",
            "- 回答時承接查詢結果與 lineage/sourceRefs；不要宣稱有 raw SQL 或可寫入 FOS。",
            "- 若查庫存，目前庫存問題優先使用 warehouse_stock_balance read model 的語意。",
            "- 回覆使用者時不要提到內部工具、MCP、hook、plugin、local service 或記憶同步細節。",
        ]
        if notes:
            lines.append("近期已學到的 FOS 圖譜脈絡：")
            lines.extend(f"- {note}" for note in notes)
        return {"context": "\n".join(lines)}

    def finalize_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        platform: str = "",
        model: str = "",
    ) -> None:
        config = _load_config()
        if not config.enabled:
            return
        observations = self.pop_pending(session_id)
        if not observations:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id or "",
            "platform": platform or "",
            "model": model or "",
            "user_message": _sanitize_text(user_message, max_chars=1200),
            "assistant_summary": _sanitize_text(assistant_response, max_chars=1600),
            "observations": observations,
        }
        self._append_record(record)
        if config.sync_enabled:
            self._sync_to_fsa_async(config, record)

    def _store_path(self) -> Path:
        return get_hermes_home() / "plugins" / "fos_graph_learning" / "learned_context.jsonl"

    def _append_record(self, record: dict[str, Any]) -> None:
        try:
            path = self._store_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            logger.debug("fos_graph_learning local append skipped: %s", exc)

    def _load_recent_notes(self, query: str, *, limit: int) -> list[str]:
        path = self._store_path()
        if not path.exists():
            return []
        query_terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query or "")
            if len(term) >= 2
        }
        notes: list[str] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-_load_config().local_history_limit:]
        except Exception:
            return []
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for obs in reversed(record.get("observations") or []):
                note = _sanitize_text(obs.get("note"), max_chars=500)
                if not note or note in notes:
                    continue
                haystack = note.lower()
                if not query_terms or any(term.lower() in haystack for term in query_terms) or len(notes) < 2:
                    notes.append(note)
                if len(notes) >= limit:
                    return notes
        return notes

    def _sync_to_fsa_async(self, config: FosGraphLearningConfig, record: dict[str, Any]) -> None:
        thread = threading.Thread(
            target=self._sync_to_fsa,
            args=(config, record),
            name="fos-graph-learning-sync",
            daemon=True,
        )
        thread.start()

    def _sync_to_fsa(self, config: FosGraphLearningConfig, record: dict[str, Any]) -> None:
        memory_text = self._build_memory_text(record)
        if not memory_text:
            return
        payload = {
            "user_message": f"FOS graph query: {record.get('user_message', '')}",
            "assistant_message": memory_text,
            "session_id": "fos_graph",
            "scope": "fos_graph",
            "sender_id": "hermes:fos_graph",
            "source_channel": FOS_SOURCE_CHANNEL,
            "metadata": {
                "trigger_source": "hermes:fos_graph",
                "async_memory_extraction": True,
                "hermes_session_id": record.get("session_id", ""),
                "platform": record.get("platform", ""),
                "task_mode": True,
                "memory_kind": "fos_graph_learning",
            },
        }
        try:
            with httpx.Client(timeout=config.fsa_timeout_seconds) as client:
                response = client.post(f"{config.fsa_base_url}/memory/ingest", json=payload)
                response.raise_for_status()
        except Exception as exc:
            logger.debug("fos_graph_learning FSA sync skipped: %s", exc)

    @staticmethod
    def _build_memory_text(record: dict[str, Any]) -> str:
        notes = [
            _sanitize_text(obs.get("note"), max_chars=600)
            for obs in record.get("observations") or []
            if isinstance(obs, dict) and obs.get("note")
        ]
        notes = [note for note in notes if note]
        if not notes:
            return ""
        lines = [
            "FOS graph 查詢學習摘要：",
            f"使用者問題：{record.get('user_message', '')}",
            "查詢觀察：",
        ]
        lines.extend(f"- {note}" for note in notes[:12])
        assistant = _sanitize_text(record.get("assistant_summary"), max_chars=800)
        if assistant:
            lines.append(f"本輪回覆重點：{assistant}")
        return "\n".join(lines)


_STATE = FosGraphLearningState()


def on_post_tool_call(**kwargs: Any) -> None:
    _STATE.record_tool_call(
        tool_name=str(kwargs.get("tool_name") or ""),
        args=kwargs.get("args") if isinstance(kwargs.get("args"), dict) else {},
        result=kwargs.get("result") or "",
        session_id=str(kwargs.get("session_id") or ""),
        duration_ms=_int_value(kwargs.get("duration_ms"), 0),
    )


def on_transform_tool_result(**kwargs: Any) -> Optional[str]:
    return _STATE.transform_result(
        tool_name=str(kwargs.get("tool_name") or ""),
        result=kwargs.get("result") or "",
    )


def on_pre_llm_call(**kwargs: Any) -> Optional[dict[str, str]]:
    return _STATE.build_context(str(kwargs.get("user_message") or ""))


def on_post_llm_call(**kwargs: Any) -> None:
    _STATE.finalize_turn(
        session_id=str(kwargs.get("session_id") or ""),
        user_message=str(kwargs.get("user_message") or ""),
        assistant_response=str(kwargs.get("assistant_response") or ""),
        platform=str(kwargs.get("platform") or ""),
        model=str(kwargs.get("model") or ""),
    )


def _reset_for_tests() -> None:
    global _STATE
    _STATE = FosGraphLearningState()


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("transform_tool_result", on_transform_tool_result)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
