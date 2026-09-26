"""Parse output markers in plain text into message components."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from astrbot.api.message_components import (
    At,
    AtAll,
    BaseMessageComponent,
    Plain,
    Reply,
)

from .config import PluginConfig

# Closed markers are replaced. An opening without a close is removed through
# the end of the current line so the raw marker is never shown to the user.
_MARKER_RE = re.compile(r"\{\{([^{}\n]*)\}\}|\{\{[^{}\n]*")
_AT_RE = re.compile(r"AT:(\d+|all)$")


class SegmentBreak:
    """Internal split point created by ``{{SEG}}``. Never sent to a platform."""


@dataclass
class ParsedText:
    """Result of parsing markers inside one plain-text segment."""

    components: list[BaseMessageComponent | SegmentBreak] = field(default_factory=list)
    force_image: bool = False
    explicit_segments: bool = False


def parse_markers(
    text: str,
    config: PluginConfig,
    *,
    reply: Reply | None,
    platform_at: bool,
    platform_reply: bool,
) -> ParsedText:
    """Replace output markers with message components.

    Args:
        text: Plain text that may contain markers.
        config: Feature switches that decide which markers take effect.
        reply: Reply component built from the triggering message.
        platform_at: Whether the current platform can send At components.
        platform_reply: Whether the current platform can send Reply components.

    Returns:
        Components in original order. Unknown, unclosed, and disabled markers
        are removed. ``force_image`` is set when ``{{IMG}}`` is enabled.
        ``explicit_segments`` is set when at least one ``{{SEG}}`` is kept.
    """
    parsed = ParsedText()
    cursor = 0
    pending = ""

    def flush() -> None:
        nonlocal pending
        if pending:
            parsed.components.append(Plain(pending))
            pending = ""

    for match in _MARKER_RE.finditer(text):
        pending += text[cursor : match.start()]
        cursor = match.end()
        token = match.group(1)
        if token is None:
            continue

        at_match = _AT_RE.fullmatch(token)
        if at_match and config.at_enable and platform_at:
            target = at_match.group(1)
            if target == "all":
                if config.allow_at_all:
                    flush()
                    parsed.components.append(AtAll())
            else:
                flush()
                parsed.components.append(At(qq=target))
            continue
        if token == "REPLY" and config.quote_enable and platform_reply and reply:
            flush()
            parsed.components.append(reply)
            continue
        if token == "SEG" and config.seg_enable:
            flush()
            parsed.components.append(SegmentBreak())
            parsed.explicit_segments = True
            continue
        if token == "IMG" and config.t2i_enable:
            parsed.force_image = True

    pending += text[cursor:]
    flush()
    return parsed
