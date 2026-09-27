"""Text cleanup and sentence splitting used before a message is sent."""

from __future__ import annotations

import re

from .config import PluginConfig


def _pair_spans(text: str, symbols: list[str]) -> list[tuple[int, int]]:
    """Find ranges protected by configured pair symbols.

    Args:
        text: Source text.
        symbols: Pair entries such as ``（）`` or ``""``.

    Returns:
        Inclusive-exclusive ranges that must not be split.
    """
    spans: list[tuple[int, int]] = []
    for symbol in symbols:
        if len(symbol) < 2:
            continue
        if symbol[0] == symbol[-1]:
            char = re.escape(symbol[0])
            pattern = re.compile(char + r".*?" + char, re.DOTALL)
        else:
            left, right = symbol[0], symbol[-1]
            pattern = re.compile(re.escape(left) + r".*?" + re.escape(right), re.DOTALL)
        spans.extend(match.span() for match in pattern.finditer(text))
    return spans


def _inside(index: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def cleanup_text(text: str, config: PluginConfig) -> str:
    """Remove configured noise from a short reply.

    Args:
        text: Combined plain text of the reply.
        config: Cleanup switches, length limit, and patterns.

    Returns:
        Cleaned text. Text at or above the length limit is returned unchanged.
    """
    if not config.cleanup_enable or len(text) >= config.cleanup_max_length:
        return text
    cleaned = text
    for pattern in config.cleanup_patterns():
        cleaned = pattern.sub("", cleaned)
    return cleaned


def split_sentences(text: str, config: PluginConfig) -> list[str]:
    """Split text on configured characters, keeping paired regions intact.

    Args:
        text: Text to split.
        config: Split characters, protected pair symbols, and tail trim list.

    Returns:
        Non-empty sentence pieces. The original text is returned when no
        split is possible.
    """
    pattern = config.split_pattern()
    if pattern is None or not text:
        return [text] if text else []

    protected = _pair_spans(text, config.seg_pair_symbols)
    pieces: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        if _inside(match.start(), protected):
            continue
        piece = trim_segment(text[start : match.end()], [], config.seg_trim_chars)
        if piece:
            pieces.append(piece)
        start = match.end()
    tail = trim_segment(text[start:], [], config.seg_trim_chars)
    if tail:
        pieces.append(tail)
    return pieces or ([text] if text else [])


def trim_segment(text: str, head: list[str], tail: list[str]) -> str:
    """Remove configured characters from both ends of one segment.

    Args:
        text: Segment text.
        head: Characters removed from the start, longest match first.
        tail: Characters removed from the end, longest match first.

    Returns:
        Text with those edge characters removed. Interior text is unchanged.
    """
    return _trim_edge(_trim_edge(text, head, from_tail=False), tail, from_tail=True)


def _trim_edge(text: str, chars: list[str], *, from_tail: bool) -> str:
    """Strip one edge, longest configured match first."""
    ordered = sorted((item for item in chars if item), key=len, reverse=True)
    changed = True
    while changed and text:
        changed = False
        for item in ordered:
            if from_tail and text.endswith(item):
                text = text[: -len(item)]
                changed = True
                break
            if not from_tail and text.startswith(item):
                text = text[len(item) :]
                changed = True
                break
    return text.strip() if not chars else text


def typing_delay(text: str, speed: float) -> float:
    """Estimate how long to wait before sending one segment.

    Args:
        text: Segment text.
        speed: Characters per second. Larger values send faster.

    Returns:
        Delay in seconds, capped so a long segment cannot stall the reply.
    """
    if speed <= 0:
        return 0.0
    return min(len(text) / speed, 8.0)
