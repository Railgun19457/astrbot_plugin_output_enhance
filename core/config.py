"""Load and normalize plugin configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger

# Platforms that can turn At / Reply / Nodes into native messages.
PLATFORM_CAPABILITIES: dict[str, frozenset[str]] = {
    "aiocqhttp": frozenset({"at", "reply", "forward"}),
    "discord": frozenset({"at", "reply"}),
    "kook": frozenset({"at", "reply"}),
    "satori": frozenset({"at", "reply", "forward"}),
    "telegram": frozenset({"at", "reply"}),
}


def supports(platform_name: str, capability: str) -> bool:
    """Return whether a platform implements a message capability.

    Args:
        platform_name: AstrBot platform type, such as ``aiocqhttp``.
        capability: One of ``at``, ``reply``, or ``forward``.

    Returns:
        True when the platform can send that component natively.
    """
    return capability in PLATFORM_CAPABILITIES.get(platform_name, frozenset())


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: object, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return [_unescape(item) for item in fallback]
    items = [_unescape(str(item)) for item in value if str(item)]
    return items or [_unescape(item) for item in fallback]


def _unescape(value: str) -> str:
    """Turn config escapes into the characters they describe.

    Args:
        value: One configured string. WebUI stores a newline entry as the two
            characters ``\\n`` rather than a line break.

    Returns:
        The string with ``\\n``, ``\\r``, and ``\\t`` decoded.
    """
    return value.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")


def _as_int(value: object, fallback: int, minimum: int = 0) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= minimum else fallback


def _as_float(value: object, fallback: float, minimum: float = 0.1) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= minimum else fallback


def _as_bool(value: object, fallback: bool) -> bool:
    return value if isinstance(value, bool) else fallback


@dataclass
class PluginConfig:
    """Normalized plugin settings used by the output pipeline."""

    llm_tool_options: set[str] = field(default_factory=set)

    prompt_enable: bool = True
    prompt_custom: str = ""

    seg_enable: bool = False
    seg_plugin_messages: bool = False
    seg_words_threshold: int = 150
    seg_sentences_threshold: int = 8
    seg_split_chars: list[str] = field(default_factory=list)
    seg_pair_symbols: list[str] = field(default_factory=list)
    seg_typing_speed: float = 8.0
    seg_trim_head_chars: list[str] = field(default_factory=list)
    seg_trim_chars: list[str] = field(default_factory=list)

    quote_enable: bool = True
    auto_quote_interval: int = 0

    at_enable: bool = True
    allow_at_all: bool = False

    forward_enable: bool = False
    forward_text_threshold: int = 500
    forward_sentences_threshold: int = 10
    default_nickname: str = "AstrBot"

    t2i_enable: bool = False
    t2i_text_threshold: int = 1000
    t2i_renderer: str = "pillow"
    pillow_template: str = ""
    font_path: str = ""
    emoji_font_path: str = ""
    auto_page: bool = True

    error_enable: bool = True
    error_keywords: list[str] = field(default_factory=list)
    plugin_error: bool = True
    user_notice: str = ""
    forward_sessions: list[str] = field(default_factory=list)

    block_enable: bool = False
    block_keywords: list[str] = field(default_factory=list)

    cleanup_enable: bool = False
    cleanup_max_length: int = 200
    cleanup_pair_symbols: list[str] = field(default_factory=list)
    cleanup_regex: list[str] = field(default_factory=list)

    def injection_text(self) -> str:
        """Build the static marker prompt appended to the system prompt.

        Returns:
            Empty string when injection is disabled, otherwise the custom
            prompt or the built-in prompt trimmed to enabled features.
        """
        if not self.prompt_enable:
            return ""
        custom = self.prompt_custom.strip()
        if custom:
            return custom

        lines = [
            "你可以用下列标记控制这条回复的呈现方式。标记写在正文里，发送前会被替换，用户看不到标记本身。未知标记会被删除。"
        ]
        if self.at_enable:
            lines.append("{{AT:数字ID}} 会 @ 对应用户。")
            if self.allow_at_all:
                lines.append("{{AT:all}} 会 @全体成员。")
        if self.quote_enable:
            lines.append("{{REPLY}} 会引用用户这条消息。")
        if self.seg_enable:
            lines.append("{{SEG}} 是分段点，前后会分成两条消息发送。")
        if self.t2i_enable:
            lines.append("{{IMG}} 会把整条回复渲染成图片。")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def cleanup_patterns(self) -> list[re.Pattern[str]]:
        """Compile cleanup patterns. Invalid expressions are skipped.

        Returns:
            Patterns that remove paired-symbol contents and custom matches.
        """
        patterns: list[re.Pattern[str]] = []
        for item in self.cleanup_pair_symbols:
            pattern = pair_placeholder_pattern(item)
            if pattern is not None:
                patterns.append(pattern)
        for item in self.cleanup_regex:
            try:
                patterns.append(re.compile(item))
            except re.error:
                logger.warning(
                    "[OutputEnhance] Invalid cleanup regex skipped: %s", item
                )
        return patterns

    def split_pattern(self) -> re.Pattern[str] | None:
        """Compile the segmented-reply splitter.

        Returns:
            A pattern that keeps the split character, or None when empty.
        """
        parts = [re.escape(item) for item in self.seg_split_chars if item]
        if not parts:
            return None
        parts.sort(key=len, reverse=True)
        return re.compile("|".join(parts))


def pair_placeholder_pattern(item: str) -> re.Pattern[str] | None:
    """Turn a ``[text]`` style placeholder into a removal pattern.

    Args:
        item: Pair symbol with the literal placeholder ``text`` in the middle.

    Returns:
        A pattern matching the wrapped content, or None when the item is not
        a placeholder.
    """
    marker = "text"
    if marker not in item:
        return None
    left, right = item.split(marker, 1)
    if not left and not right:
        return None
    return re.compile(re.escape(left) + r".*?" + re.escape(right), re.DOTALL)


def load_config(raw: dict[str, Any] | None) -> PluginConfig:
    """Normalize the raw AstrBot config dict.

    Args:
        raw: Plugin config passed to the plugin constructor.

    Returns:
        A config object with defaults filled in for missing or invalid values.
    """
    data = raw or {}
    tools = data.get("llm_tool_options", ["send_forward_message", "send_text_as_image"])
    tool_options = (
        {str(item) for item in tools if str(item)} if isinstance(tools, list) else set()
    )

    prompt = _as_dict(data.get("prompt_injection"))
    segmented = _as_dict(data.get("segmented_reply"))
    quote = _as_dict(data.get("quote_reply"))
    at_parse = _as_dict(data.get("at_parse"))
    forward = _as_dict(data.get("forward"))
    text_to_image = _as_dict(data.get("text_to_image"))
    error = _as_dict(data.get("error_intercept"))
    block = _as_dict(data.get("message_block"))
    cleanup = _as_dict(data.get("text_cleanup"))

    return PluginConfig(
        llm_tool_options=tool_options,
        prompt_enable=_as_bool(prompt.get("enable"), True),
        prompt_custom=str(prompt.get("prompt") or ""),
        seg_enable=_as_bool(segmented.get("enable"), True),
        seg_plugin_messages=_as_bool(segmented.get("plugin_messages"), False),
        seg_words_threshold=_as_int(segmented.get("words_threshold"), 150, 1),
        seg_sentences_threshold=_as_int(segmented.get("sentences_threshold"), 8, 1),
        seg_split_chars=_as_str_list(
            segmented.get("split_chars"), ["。", "？", "！", "……", "~", "\\n"]
        ),
        seg_pair_symbols=_as_str_list(
            segmented.get("pair_symbols"),
            [
                '""',
                "''",
                "（）",
                "()",
                "【】",
                "[]",
                "{}",
                "《》",
                "「」",
                "『』",
                "“”",
                "‘’",
            ],
        ),
        seg_typing_speed=_as_float(segmented.get("typing_speed"), 30.0),
        seg_trim_head_chars=_as_str_list(segmented.get("trim_head_chars"), []),
        seg_trim_chars=_as_str_list(
            segmented.get("trim_chars"),
            ["。", "，", "；", "、", ",", ".", ";"],
        ),
        quote_enable=_as_bool(quote.get("enable"), True),
        auto_quote_interval=_as_int(quote.get("auto_quote_interval"), 2),
        at_enable=_as_bool(at_parse.get("enable"), True),
        allow_at_all=_as_bool(at_parse.get("allow_at_all"), False),
        forward_enable=_as_bool(forward.get("enable"), True),
        forward_text_threshold=_as_int(forward.get("text_threshold"), 300, 1),
        forward_sentences_threshold=_as_int(forward.get("sentences_threshold"), 10, 1),
        default_nickname=str(forward.get("default_nickname") or "AstrBot"),
        t2i_enable=_as_bool(text_to_image.get("enable"), False),
        t2i_text_threshold=_as_int(text_to_image.get("text_threshold"), 1000, 1),
        t2i_renderer=(
            "astrbot_t2i"
            if text_to_image.get("renderer") == "astrbot_t2i"
            else "pillow"
        ),
        pillow_template=str(
            text_to_image.get("pillow_template") or "templates/light.json"
        ),
        font_path=str(
            text_to_image.get("font_path")
            or "https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf"
        ),
        emoji_font_path=str(
            text_to_image.get("emoji_font_path")
            or "https://cdn.jsdelivr.net/gh/googlefonts/noto-emoji@main/2D/fonts/NotoColorEmoji.ttf"
        ),
        auto_page=_as_bool(text_to_image.get("auto_page"), True),
        error_enable=_as_bool(error.get("enable"), True),
        error_keywords=_as_str_list(
            error.get("keywords"),
            [
                "Traceback (most recent call last)",
                "Error occurred while processing agent request",
                "在调用插件",
                "LLM 请求失败",
                "API 调用失败",
            ],
        ),
        plugin_error=_as_bool(error.get("plugin_error"), True),
        user_notice=str(error.get("user_notice") or ""),
        forward_sessions=[
            item.strip()
            for item in _as_str_list(error.get("forward_sessions"), [])
            if item.strip()
        ],
        block_enable=_as_bool(block.get("enable"), False),
        block_keywords=[
            item for item in _as_str_list(block.get("keywords"), []) if item
        ],
        cleanup_enable=_as_bool(cleanup.get("enable"), False),
        cleanup_max_length=_as_int(cleanup.get("max_length"), 200, 1),
        cleanup_pair_symbols=_as_str_list(
            cleanup.get("pair_symbols"),
            ["[text]", "(text)", "（text）", "【text】", "&&text&&"],
        ),
        cleanup_regex=_as_str_list(cleanup.get("regex"), []),
    )
