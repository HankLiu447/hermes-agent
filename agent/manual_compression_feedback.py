"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
) -> dict[str, Any]:
    """Return consistent user-facing feedback for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)

    if noop:
        headline = f"壓縮後沒有變更：{before_count} 則訊息"
        if after_tokens == before_tokens:
            token_line = (
                f"預估請求大小：~{before_tokens:,} tokens（未變更）"
            )
        else:
            token_line = (
                f"預估請求大小：~{before_tokens:,} → "
                f"~{after_tokens:,} tokens"
            )
    else:
        headline = f"已壓縮：{before_count} → {after_count} 則訊息"
        token_line = (
            f"預估請求大小：~{before_tokens:,} → "
            f"~{after_tokens:,} tokens"
        )

    note = None
    if not noop and after_count < before_count and after_tokens > before_tokens:
        note = (
            "注意：訊息數變少後，預估 token 仍可能上升，因為壓縮會把對話改寫成更密集的摘要。"
        )

    return {
        "noop": noop,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
