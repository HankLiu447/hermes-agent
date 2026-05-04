"""FSA/Xiaoxi memory provider for Hermes task-mode conversations.

The provider reads a safe subset of the local FlySuiteAgent APIs before a
Hermes turn and mirrors completed turns back after delivery. It deliberately
does not connect to the database, send proactive messages, or expose tools.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from agent.memory_manager import sanitize_context
from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://127.0.0.1:18793"
DEFAULT_CONTEXT_HOURS = 720
DEFAULT_CONTEXT_LIMIT = 8
DEFAULT_ONTOLOGY_LIMIT = 6
DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_SYNC_SCOPE = "codex"
DEFAULT_SOURCE_CHANNEL = "hermes_line"

_MAX_ITEM_CHARS = 900
_MAX_SYNC_CHARS = 8000

_SENSITIVE_LINE_RE = re.compile(
    r"("
    r"健康|睡眠|精力|壓力|恢復|心率|血氧|位置|定位|座標|lat\s*=|lon\s*=|"
    r"vision|camera|screen|desktop|looki|health|sleep|stress|location|"
    r"dmn|dream|temporal|plaud|life_state|immediate_narrative|"
    r"認知迴圈|深度休眠|DeepSleep|SILENT"
    r")",
    re.IGNORECASE,
)

_INTERNAL_REPLACEMENTS = (
    (re.compile(r"\bFSA(?:_[A-Z0-9]+)+\b"), "系統環境變數"),
    (re.compile(r"\bFlySuiteAgent\b|(?<![A-Za-z0-9])FSA(?![A-Za-z0-9])"), "主要聊天系統"),
    (re.compile(r"\bHermes\b", re.IGNORECASE), "任務工作區"),
    (re.compile(r"\bS1\b"), "長期互動側"),
    (re.compile(r"\bS2\b"), "任務執行側"),
    (re.compile(r"\bmemory_core\b", re.IGNORECASE), "長期脈絡"),
    (re.compile(r"\bontology\b", re.IGNORECASE), "知識背景"),
    (re.compile(r"\bhydrate\b", re.IGNORECASE), "上下文整理"),
    (re.compile(r"\bsystem prompt\b", re.IGNORECASE), "系統指示"),
    (re.compile(r"\bbridge\b", re.IGNORECASE), "銜接流程"),
    (re.compile(r"http://127\.0\.0\.1:\d+(?:/[^\s，。)]*)?"), "本機服務"),
    (re.compile(r"http://localhost:\d+(?:/[^\s，。)]*)?"), "本機服務"),
)


@dataclass
class FsaXiaoxiConfig:
    enabled: bool = True
    base_url: str = DEFAULT_BASE_URL
    context_hours: int = DEFAULT_CONTEXT_HOURS
    context_limit: int = DEFAULT_CONTEXT_LIMIT
    ontology_limit: int = DEFAULT_ONTOLOGY_LIMIT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    sync_scope: str = DEFAULT_SYNC_SCOPE
    source_channel: str = DEFAULT_SOURCE_CHANNEL
    hank_line_user_ids: set[str] = field(default_factory=set)
    private_context_policy: str = "hank_dm"


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


def _split_ids(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _cfg_get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    section = cfg.get("fsa_xiaoxi") if isinstance(cfg, dict) else {}
    if not isinstance(section, dict):
        return default
    return section.get(key, default)


def _load_config() -> FsaXiaoxiConfig:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        cfg = {}

    enabled = _truthy(
        os.getenv("FSA_XIAOXI_ENABLED", _cfg_get(cfg, "enabled")),
        default=True,
    )
    base_url = str(
        os.getenv("FSA_XIAOXI_BASE_URL")
        or _cfg_get(cfg, "base_url")
        or DEFAULT_BASE_URL
    ).rstrip("/")
    context_hours = _int_value(
        os.getenv("FSA_XIAOXI_CONTEXT_HOURS", _cfg_get(cfg, "context_hours")),
        DEFAULT_CONTEXT_HOURS,
    )
    context_limit = _int_value(
        os.getenv("FSA_XIAOXI_CONTEXT_LIMIT", _cfg_get(cfg, "context_limit")),
        DEFAULT_CONTEXT_LIMIT,
    )
    ontology_limit = _int_value(
        os.getenv("FSA_XIAOXI_ONTOLOGY_LIMIT", _cfg_get(cfg, "ontology_limit")),
        DEFAULT_ONTOLOGY_LIMIT,
    )
    timeout_seconds = _float_value(
        os.getenv("FSA_XIAOXI_TIMEOUT_SECONDS", _cfg_get(cfg, "timeout_seconds")),
        DEFAULT_TIMEOUT_SECONDS,
    )
    sync_scope = str(_cfg_get(cfg, "sync_scope", DEFAULT_SYNC_SCOPE) or DEFAULT_SYNC_SCOPE)
    source_channel = str(
        _cfg_get(cfg, "source_channel", DEFAULT_SOURCE_CHANNEL) or DEFAULT_SOURCE_CHANNEL
    )
    hank_ids = (
        _split_ids(os.getenv("FSA_XIAOXI_HANK_LINE_USER_IDS"))
        or _split_ids(_cfg_get(cfg, "hank_line_user_ids"))
        or _split_ids(_cfg_get(cfg, "hank_line_user_id"))
    )
    policy = str(_cfg_get(cfg, "private_context_policy", "hank_dm") or "hank_dm").strip().lower()
    return FsaXiaoxiConfig(
        enabled=enabled,
        base_url=base_url,
        context_hours=context_hours,
        context_limit=max(1, context_limit),
        ontology_limit=max(1, ontology_limit),
        timeout_seconds=max(0.2, timeout_seconds),
        sync_scope=sync_scope,
        source_channel=source_channel,
        hank_line_user_ids=hank_ids,
        private_context_policy=policy,
    )


def _clean_text(value: Any, *, max_chars: int = _MAX_ITEM_CHARS, drop_sensitive: bool = True) -> str:
    text = sanitize_context(str(value or ""))
    text = text.replace("\x00", " ")
    text = re.sub(r"<[^>\n]{1,80}>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if drop_sensitive and _SENSITIVE_LINE_RE.search(text):
        return ""
    for pattern, replacement in _INTERNAL_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email]", text)
    text = re.sub(r"(sk|xox[baprs]|ghp|gho|line)[-_A-Za-z0-9]{16,}", "[secret]", text)
    text = re.sub(r"\s+", " ", text).strip(" -:：")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _append_line(lines: list[str], value: str) -> None:
    value = value.strip()
    if value and value not in lines:
        lines.append(value)


class FsaXiaoxiMemoryProvider(MemoryProvider):
    """Context-only memory provider backed by the local FSA safe APIs."""

    def __init__(self, config: Optional[FsaXiaoxiConfig] = None):
        self._config = config or _load_config()
        self._session_id = ""
        self._platform = ""
        self._user_id = ""
        self._user_name = ""
        self._chat_id = ""
        self._chat_type = ""
        self._gateway_session_key = ""
        self._last_sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "fsa_xiaoxi"

    def is_available(self) -> bool:
        return bool(self._config.enabled and self._config.base_url)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or "")
        self._platform = str(kwargs.get("platform") or "")
        self._user_id = str(kwargs.get("user_id") or "")
        self._user_name = str(kwargs.get("user_name") or "")
        self._chat_id = str(kwargs.get("chat_id") or "")
        self._chat_type = str(kwargs.get("chat_type") or "")
        self._gateway_session_key = str(kwargs.get("gateway_session_key") or "")

    def system_prompt_block(self) -> str:
        return (
            "When shared context is present, use it only as private background. "
            "Do not reveal the context source, internal architecture, storage names, "
            "local URLs, or service boundaries to the user."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self.is_available() or not self._should_inject_private_context():
            return ""
        clean_query = _clean_text(query, max_chars=1200, drop_sensitive=False)
        if not clean_query:
            return ""

        try:
            memories = self._post_json(
                "/memory/codex-context",
                {
                    "query": clean_query,
                    "hours": self._config.context_hours,
                    "limit": self._config.context_limit,
                    "include_global": True,
                },
            )
            facts = self._post_json(
                "/ontology/recall",
                {"query": clean_query, "limit": self._config.ontology_limit},
            )
            state = {
                "identity": self._get_json("/state/identity"),
                "active_context": self._get_json("/state/active_context"),
                "open_tasks": self._get_json("/state/open_tasks"),
                "immediate_narrative": self._get_json("/state/immediate_narrative"),
            }
        except Exception as exc:
            logger.debug("fsa_xiaoxi prefetch skipped: %s", exc)
            return ""

        return self._format_context(memories, facts, state)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self.is_available() or not user_content or not assistant_content:
            return
        payload = self._build_sync_payload(user_content, assistant_content, session_id=session_id)

        def _worker() -> None:
            try:
                self._post_json("/memory/ingest", payload)
            except Exception as exc:
                logger.debug("fsa_xiaoxi sync skipped: %s", exc)

        thread = threading.Thread(target=_worker, name="fsa-xiaoxi-sync", daemon=True)
        self._last_sync_thread = thread
        thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        thread = self._last_sync_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    def _request_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        url = f"{self._config.base_url}{path}"
        with httpx.Client(timeout=self._config.timeout_seconds) as client:
            if method == "GET":
                response = client.get(url)
            else:
                response = client.post(url, json=payload or {})
            response.raise_for_status()
            if not response.content:
                return None
            return response.json()

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request_json("POST", path, payload)

    def _get_json(self, path: str) -> Any:
        return self._request_json("GET", path)

    def _should_inject_private_context(self) -> bool:
        if self._platform and self._platform.lower() != "line":
            return False
        if (self._chat_type or "").lower() != "dm":
            return False
        policy = self._config.private_context_policy
        if policy in {"dm", "line_dm"}:
            return True
        if not self._config.hank_line_user_ids:
            return False
        candidates = {self._user_id, self._chat_id}
        return bool(candidates & self._config.hank_line_user_ids)

    def _memory_session_id(self) -> str:
        if self._should_inject_private_context():
            return "main"
        if self._platform == "line" and self._chat_id:
            return f"hermes:line:{self._chat_id}"
        if self._platform and self._chat_id:
            return f"hermes:{self._platform}:{self._chat_id}"
        return self._session_id or "hermes:unknown"

    def _format_context(self, memories: Any, facts: Any, state: dict[str, Any]) -> str:
        recent_lines = self._extract_memory_lines(memories)
        fact_lines = self._extract_fact_lines(facts)
        task_lines = self._extract_state_lines(state)

        parts: list[str] = []
        if recent_lines:
            parts.append("共享近況：\n" + "\n".join(f"- {line}" for line in recent_lines[: self._config.context_limit]))
        if task_lines:
            parts.append("任務脈絡：\n" + "\n".join(f"- {line}" for line in task_lines[:6]))
        if fact_lines:
            parts.append("相關背景：\n" + "\n".join(f"- {line}" for line in fact_lines[: self._config.ontology_limit]))
        if not parts:
            return ""
        parts.append("回覆時自然承接即可，不要提到背景來源、內部架構、系統名稱或本機服務資訊。")
        return "\n\n".join(parts)

    def _extract_memory_lines(self, payload: Any) -> list[str]:
        lines: list[str] = []
        if not isinstance(payload, dict):
            return lines
        for item in payload.get("memories") or []:
            if not isinstance(item, dict):
                continue
            content = _clean_text(item.get("content"), drop_sensitive=True)
            _append_line(lines, content)
        return lines

    def _extract_fact_lines(self, payload: Any) -> list[str]:
        lines: list[str] = []
        if not isinstance(payload, dict):
            return lines
        formatted = _clean_text(payload.get("formatted"), drop_sensitive=True)
        if formatted:
            for part in re.split(r"\s*-\s+", formatted):
                _append_line(lines, _clean_text(part, drop_sensitive=True))
        for fact in payload.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            subject = _clean_text(fact.get("subject_name"), max_chars=120)
            predicate = _clean_text(fact.get("predicate_name"), max_chars=80)
            obj = _clean_text(fact.get("object_name") or fact.get("value"), max_chars=160)
            if subject and predicate and obj:
                _append_line(lines, f"{subject} {predicate} {obj}")
        return lines

    def _extract_state_lines(self, state: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        identity = state.get("identity")
        if isinstance(identity, dict):
            tone = _clean_text(identity.get("tone"), max_chars=80)
            mood = _clean_text(identity.get("mood"), max_chars=80)
            if tone or mood:
                _append_line(lines, f"目前語氣偏向 {tone or mood}")

        active = state.get("active_context")
        if isinstance(active, dict):
            for key in ("topic", "summary"):
                value = _clean_text(active.get(key), drop_sensitive=True)
                _append_line(lines, value)

        tasks = state.get("open_tasks")
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict):
                    desc = _clean_text(task.get("desc") or task.get("title"), drop_sensitive=True)
                    status = _clean_text(task.get("status"), max_chars=80)
                    if desc:
                        _append_line(lines, f"{desc}" + (f"（{status}）" if status else ""))
                else:
                    _append_line(lines, _clean_text(task, drop_sensitive=True))

        narrative = state.get("immediate_narrative")
        if isinstance(narrative, list):
            for item in narrative[:3]:
                _append_line(lines, _clean_text(item, drop_sensitive=True))
        return lines

    def _build_sync_payload(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> dict[str, Any]:
        user_message = _clean_text(user_content, max_chars=_MAX_SYNC_CHARS, drop_sensitive=False)
        assistant_message = _clean_text(assistant_content, max_chars=_MAX_SYNC_CHARS, drop_sensitive=False)
        sender_id = f"line:{self._user_id}" if self._user_id else ""
        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "session_id": self._memory_session_id(),
            "scope": self._config.sync_scope,
            "sender_id": sender_id,
            "source_channel": self._config.source_channel,
            "metadata": {
                "trigger_source": "hermes:line",
                "async_memory_extraction": True,
                "hermes_session_key": self._gateway_session_key,
                "hermes_session_id": session_id or self._session_id,
                "platform": self._platform,
                "chat_type": self._chat_type,
                "task_mode": True,
            },
        }


def register(ctx) -> None:
    ctx.register_memory_provider(FsaXiaoxiMemoryProvider())
