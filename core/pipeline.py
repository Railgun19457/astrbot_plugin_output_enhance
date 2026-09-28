"""Turn a finished reply chain into the chain that should be sent."""

from __future__ import annotations

from dataclasses import replace

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import (
    At,
    AtAll,
    BaseMessageComponent,
    Face,
    File,
    Image,
    Node,
    Nodes,
    Plain,
    Record,
    Reply,
    Video,
)

from .config import PluginConfig, supports
from .markers import SegmentBreak, parse_markers
from .text_ops import cleanup_text, split_sentences, trim_segment

_TEXT_TYPES = (Plain, At, AtAll, Reply)
# Components a forward node can carry without dropping them.
_FORWARD_CONTENT = (Plain, At, AtAll, Image, Record, Video, File, Face)


def _plain_text(components: list[BaseMessageComponent | SegmentBreak]) -> str:
    return "".join(comp.text for comp in components if isinstance(comp, Plain))


def _only_text(components: list[BaseMessageComponent | SegmentBreak]) -> bool:
    return all(isinstance(comp, (*_TEXT_TYPES, SegmentBreak)) for comp in components)


def _without_breaks(
    components: list[BaseMessageComponent | SegmentBreak],
) -> list[BaseMessageComponent]:
    """Drop segment breaks and the newlines models wrap around them.

    Args:
        components: Chain that may contain internal segment breaks.

    Returns:
        Components with the breaks removed. Newlines immediately before and
        after a break are removed so an unsplit reply does not keep a blank
        line in place of ``{{SEG}}``.
    """
    cleaned: list[BaseMessageComponent] = []
    after_break = False
    for comp in components:
        if isinstance(comp, SegmentBreak):
            if cleaned and isinstance(cleaned[-1], Plain):
                text = trim_segment(cleaned[-1].text, [], ["\n"])
                if text:
                    cleaned[-1] = Plain(text)
                else:
                    cleaned.pop()
            after_break = True
            continue
        if after_break and isinstance(comp, Plain):
            text = trim_segment(comp.text, ["\n"], [])
            after_break = False
            if text:
                cleaned.append(Plain(text))
            continue
        after_break = False
        cleaned.append(comp)
    return cleaned


def _build_reply(event: AstrMessageEvent) -> Reply | None:
    message_id = getattr(event.message_obj, "message_id", None)
    if message_id in (None, ""):
        return None
    return Reply(
        id=message_id,
        sender_id=event.get_sender_id(),
        sender_nickname=event.get_sender_name(),
        message_str=event.message_str,
    )


async def _message_gap(event: AstrMessageEvent) -> int | None:
    """Count other people's messages stored after the triggering message.

    The current row is AstrBot's persisted history id, not the platform
    message id. Rows inserted for the bot itself do not count.

    Args:
        event: Event being replied to.

    Returns:
        The number of newer messages from other senders. None when persisted
        history or the current row is unavailable.
    """
    current_id = event.get_extra("_current_platform_message_history_id")
    context = getattr(event, "context", None)
    manager = getattr(context, "message_history_manager", None)
    if not isinstance(current_id, int) or current_id <= 0 or manager is None:
        return None
    try:
        history = await manager.get(
            platform_id=event.get_platform_id(),
            user_id=event.unified_msg_origin,
            page_size=50,
        )
    except Exception:  # noqa: BLE001
        logger.exception("[OutputEnhance] Failed to read message history.")
        return None
    self_id = str(event.get_self_id() or "")
    return sum(
        1
        for record in history
        if isinstance(getattr(record, "id", None), int)
        and record.id > current_id
        and str(getattr(record, "sender_id", "") or "") != self_id
    )


async def prepare_chain(
    event: AstrMessageEvent,
    chain: list[BaseMessageComponent],
    config: PluginConfig,
) -> tuple[list[list[BaseMessageComponent]], bool]:
    """Parse markers and apply passive text rules.

    Args:
        event: Event being replied to.
        chain: Chain prepared by the framework before decoration.
        config: Normalized plugin config.

    Returns:
        Message groups to send in order, and whether image rendering should
        replace the whole text reply. Failures are handled by the caller.
    """
    # Disable segmentation before parsing so {{SEG}} is removed instead of
    # becoming a split when this reply is not eligible.
    if not _segment_this_result(event, config):
        config = replace(config, seg_enable=False)

    platform = event.get_platform_name()
    platform_at = supports(platform, "at")
    platform_reply = supports(platform, "reply")
    reply = _build_reply(event) if platform_reply else None

    converted: list[BaseMessageComponent | SegmentBreak] = []
    force_image = False
    explicit_segments = False
    for comp in chain:
        if not isinstance(comp, Plain):
            converted.append(comp)
            continue
        parsed = parse_markers(
            comp.text,
            config,
            reply=reply,
            platform_at=platform_at,
            platform_reply=platform_reply,
        )
        converted.extend(parsed.components)
        force_image = force_image or parsed.force_image
        explicit_segments = explicit_segments or parsed.explicit_segments

    if config.cleanup_enable and _only_text(converted):
        cleaned = cleanup_text(_plain_text(converted), config)
        if cleaned != _plain_text(converted):
            converted = _restore_cleaned_text(converted, cleaned)

    plain = _plain_text(converted)
    has_reply = any(isinstance(comp, Reply) for comp in converted)
    gap = await _message_gap(event)
    if (
        config.quote_enable
        and platform_reply
        and reply is not None
        and not has_reply
        and config.auto_quote_interval > 0
        and gap is not None
        and gap >= config.auto_quote_interval
    ):
        converted.insert(0, reply)

    if explicit_segments:
        groups = _group_segments(converted, config)
        # The word threshold only suppresses automatic splitting. An explicit
        # marker stays split until there are too many messages to send, and
        # then only a forward can preserve one node per marker.
        if len(groups) > 1 and (
            len(groups) <= config.seg_sentences_threshold
            or (
                config.forward_enable
                and supports(platform, "forward")
                and len(groups) > config.forward_sentences_threshold
            )
        ):
            return groups, force_image
        return [_without_breaks(converted)], force_image

    if not _only_text(converted) or _exceeds_passive_threshold(plain, config):
        return [_without_breaks(converted)], force_image

    candidates = _candidate_segments(converted, config)
    if candidates and _within_segment_limits(plain, len(candidates), config):
        return candidates, force_image
    # Too many messages to send separately. Keep the splits only when the
    # forward sentence limit also trips, so each split becomes one node
    # instead of being flattened into a single forward message.
    if (
        candidates
        and config.forward_enable
        and supports(platform, "forward")
        and len(candidates) > config.forward_sentences_threshold
    ):
        return candidates, force_image

    return [_without_breaks(converted)], force_image


def _restore_cleaned_text(
    components: list[BaseMessageComponent | SegmentBreak],
    cleaned: str,
) -> list[BaseMessageComponent | SegmentBreak]:
    """Put cleaned plain text back where its original pieces were.

    Args:
        components: Chain containing the original plain pieces and components
            that must keep their relative positions.
        cleaned: Plain text after cleanup removed a span.

    Returns:
        The chain with each original plain span replaced by the portion of
        the cleaned text that still occupies it. Mentions, replies, and
        segment breaks stay in place. A span whose text was entirely removed
        contributes no plain component.
    """
    original = _plain_text(components)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for comp in components:
        if isinstance(comp, Plain):
            spans.append((cursor, cursor + len(comp.text)))
            cursor += len(comp.text)
    if len(cleaned) > len(original):
        return components
    removals: list[tuple[int, int]] = []
    cleaned_index = 0
    start = 0
    while start < len(original):
        if cleaned_index < len(cleaned) and original[start] == cleaned[cleaned_index]:
            cleaned_index += 1
            start += 1
            continue
        end = start + 1
        while end < len(original) and (
            cleaned_index >= len(cleaned) or original[end] != cleaned[cleaned_index]
        ):
            end += 1
        removals.append((start, end))
        start = end

    restored: list[tuple[Plain, str]] = []
    for comp, span in zip(
        (comp for comp in components if isinstance(comp, Plain)),
        spans,
    ):
        piece = comp.text
        for removal_start, removal_end in reversed(removals):
            start = max(span[0], removal_start) - span[0]
            end = min(span[1], removal_end) - span[0]
            if start < end:
                piece = piece[:start] + piece[end:]
        if piece:
            restored.append((comp, piece))
    pieces = iter(restored)

    converted: list[BaseMessageComponent | SegmentBreak] = []
    for comp in components:
        if not isinstance(comp, Plain):
            converted.append(comp)
            continue
        replacement = next(pieces, None)
        if replacement is not None and replacement[0] is comp:
            converted.append(Plain(replacement[1]))
    return converted


def _segment_this_result(event: AstrMessageEvent, config: PluginConfig) -> bool:
    """Return whether this reply should be split into multiple messages.

    Args:
        event: Event being replied to.
        config: Segment switches.

    Returns:
        False when segmentation is off, or when the message comes from another
        plugin and plugin-message segmentation is disabled. Model replies,
        including agent runner errors, still follow the main segment switch.
    """
    if not config.seg_enable:
        return False
    if config.seg_plugin_messages:
        return True
    result = event.get_result()
    return result is not None and result.is_model_result()


def _candidate_segments(
    components: list[BaseMessageComponent | SegmentBreak],
    config: PluginConfig,
) -> list[list[BaseMessageComponent]] | None:
    """Build automatic sentence splits before either limit rejects them.

    Args:
        components: Chain after marker parsing.
        config: Segment switches and split rules.

    Returns:
        One group per sentence, or None when the reply is too long or cannot
        be split. Explicit ``{{SEG}}`` groups are built by the caller.
    """
    if not config.seg_enable:
        return None

    plain = _plain_text(components)
    if not plain or len(plain) >= config.seg_words_threshold:
        return None
    sentences = split_sentences(plain, config)
    if len(sentences) <= 1:
        return None
    prefix = [comp for comp in components if not isinstance(comp, Plain | SegmentBreak)]
    return [prefix + [Plain(sentence)] for sentence in sentences]


def _within_segment_limits(text: str, count: int, config: PluginConfig) -> bool:
    """Return whether this many splits may be sent as separate messages."""
    return (
        len(text) < config.seg_words_threshold
        and count <= config.seg_sentences_threshold
    )


def _exceeds_passive_threshold(text: str, config: PluginConfig) -> bool:
    """Keep long replies whole so image and forward thresholds can see them."""
    if config.t2i_enable and len(text) >= config.t2i_text_threshold:
        return True
    return config.forward_enable and len(text) >= config.forward_text_threshold


def _group_segments(
    components: list[BaseMessageComponent | SegmentBreak],
    config: PluginConfig,
) -> list[list[BaseMessageComponent]]:
    """Group components around explicit ``{{SEG}}`` breaks.

    Args:
        components: Chain containing internal segment breaks.
        config: Characters removed from the edges of each segment.

    Returns:
        One component group for each message that should be sent separately.
        Newlines wrapped around a marker are removed with the configured
        head and tail characters.
    """
    groups: list[list[BaseMessageComponent]] = []
    current: list[BaseMessageComponent] = []
    for comp in components:
        if isinstance(comp, SegmentBreak):
            if current:
                groups.append(current)
                current = []
            continue
        current.append(comp)
    if current:
        groups.append(current)
    trimmed = [_trim_group(group, config) for group in groups]
    return [group for group in trimmed if group] or [[]]


def _trim_group(
    group: list[BaseMessageComponent],
    config: PluginConfig,
) -> list[BaseMessageComponent]:
    """Trim the outer text edges of one explicit segment."""
    indexes = [index for index, comp in enumerate(group) if isinstance(comp, Plain)]
    if not indexes:
        return group
    trimmed = list(group)
    first, last = indexes[0], indexes[-1]
    if first == last:
        text = trim_segment(
            trimmed[first].text,
            config.seg_trim_head_chars,
            config.seg_trim_chars,
        )
        trimmed[first] = Plain(text) if text else None
    else:
        head = trim_segment(trimmed[first].text, config.seg_trim_head_chars, [])
        tail = trim_segment(trimmed[last].text, [], config.seg_trim_chars)
        trimmed[first] = Plain(head) if head else None
        trimmed[last] = Plain(tail) if tail else None
    return [comp for comp in trimmed if comp is not None]


def should_forward(
    chain: list[BaseMessageComponent],
    config: PluginConfig,
    platform: str,
) -> bool:
    """Decide whether a plain reply should become one forward message.

    Args:
        chain: Chain after marker parsing and segmentation.
        config: Forward switches and thresholds.
        platform: Current platform type.

    Returns:
        True when the platform supports forwards and a threshold is reached.
        Image rendering takes priority and is decided by the caller.
    """
    if not config.forward_enable or not supports(platform, "forward"):
        return False
    if not all(isinstance(comp, (*_FORWARD_CONTENT, Reply)) for comp in chain):
        return False
    text = _plain_text(chain)
    sentence_count = sum(isinstance(comp, Plain) for comp in chain)
    return (
        len(text) >= config.forward_text_threshold
        or sentence_count > config.forward_sentences_threshold
    )


def should_render_image(
    chain: list[BaseMessageComponent],
    config: PluginConfig,
    force_image: bool,
) -> bool:
    """Decide whether the plain reply should be rendered as an image.

    Args:
        chain: Chain after marker parsing and segmentation.
        config: Image switches and the length threshold.
        force_image: Whether the model emitted ``{{IMG}}``.

    Returns:
        True when the chain is text-only and image output is requested.
    """
    if not config.t2i_enable or not _only_text(chain):
        return False
    if any(isinstance(comp, (At, Reply)) for comp in chain):
        return False
    if force_image:
        return True
    return len(_plain_text(chain)) >= config.t2i_text_threshold


def to_nodes(
    groups: list[list[BaseMessageComponent]],
    nickname: str,
) -> list[Nodes]:
    """Wrap each segment into its own forward node.

    Args:
        groups: Segment groups. One group becomes one message inside the
            forward. A reply that was never split is a single group.
        nickname: Display name used for every node.

    Returns:
        A single ``Nodes`` component. Text, images, files, audio, video,
        emoji, and mentions are kept in order. The sender id is ``0`` so QQ
        keeps the custom nickname instead of the bot's group card.
    """
    nodes: list[Node] = []
    for group in groups:
        content: list[BaseMessageComponent] = []
        for comp in group:
            if isinstance(comp, Plain):
                if comp.text.strip():
                    content.append(comp)
            elif isinstance(comp, _FORWARD_CONTENT):
                content.append(comp)
        if content:
            nodes.append(Node(name=nickname, uin="0", content=content))
    if not nodes:
        nodes.append(Node(name=nickname, uin="0", content=[Plain("")]))
    return [Nodes(nodes)]


def image_component(path_or_url: str) -> Image:
    """Build an image component from a renderer result.

    Args:
        path_or_url: Remote URL or local file path.

    Returns:
        An image component the platform adapter can send.
    """
    if path_or_url.startswith(("http://", "https://")):
        return Image.fromURL(path_or_url)
    return Image.fromFileSystem(path_or_url)
