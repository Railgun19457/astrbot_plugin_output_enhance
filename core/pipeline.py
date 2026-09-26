"""Turn a finished reply chain into the chain that should be sent."""

from __future__ import annotations

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import (
    At,
    BaseMessageComponent,
    Image,
    Node,
    Nodes,
    Plain,
    Reply,
)

from .config import PluginConfig, supports
from .markers import SegmentBreak, parse_markers
from .text_ops import cleanup_text, split_sentences

_TEXT_TYPES = (Plain, At, Reply)


def _plain_text(components: list[BaseMessageComponent | SegmentBreak]) -> str:
    return "".join(comp.text for comp in components if isinstance(comp, Plain))


def _only_text(components: list[BaseMessageComponent | SegmentBreak]) -> bool:
    return all(isinstance(comp, (*_TEXT_TYPES, SegmentBreak)) for comp in components)


def _without_breaks(
    components: list[BaseMessageComponent | SegmentBreak],
) -> list[BaseMessageComponent]:
    return [comp for comp in components if not isinstance(comp, SegmentBreak)]


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


def _message_gap(event: AstrMessageEvent) -> int | None:
    """Count messages that arrived after the triggering message.

    Returns:
        The gap when history is available, otherwise None.
    """
    history = getattr(event, "platform_message_history", None)
    current_id = getattr(event.message_obj, "message_id", None)
    if not isinstance(history, list) or current_id in (None, ""):
        return None
    ids = [getattr(message, "message_id", None) for message in history]
    if current_id not in ids:
        return None
    return len(ids) - ids.index(current_id) - 1


def prepare_chain(
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
            preserved = [comp for comp in converted if not isinstance(comp, Plain)]
            converted = [*preserved, Plain(cleaned)] if cleaned else preserved

    plain = _plain_text(converted)
    has_reply = any(isinstance(comp, Reply) for comp in converted)
    gap = _message_gap(event)
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

    if not _only_text(converted) or _exceeds_passive_threshold(plain, config):
        return [_without_breaks(converted)], force_image

    if explicit_segments and config.seg_enable:
        return _group_segments(converted), force_image

    if config.seg_enable and plain and len(plain) < config.seg_words_threshold:
        sentences = split_sentences(plain, config)
        if 1 < len(sentences) <= config.seg_sentences_threshold:
            prefix = [
                comp for comp in converted if not isinstance(comp, Plain | SegmentBreak)
            ]
            return [prefix + [Plain(sentence)] for sentence in sentences], force_image

    return [_without_breaks(converted)], force_image


def _exceeds_passive_threshold(text: str, config: PluginConfig) -> bool:
    """Keep long replies whole so image and forward thresholds can see them."""
    if config.t2i_enable and len(text) >= config.t2i_text_threshold:
        return True
    return config.forward_enable and len(text) >= config.forward_text_threshold


def _group_segments(
    components: list[BaseMessageComponent | SegmentBreak],
) -> list[list[BaseMessageComponent]]:
    """Group components around explicit ``{{SEG}}`` breaks.

    Args:
        components: Chain containing internal segment breaks.

    Returns:
        One component group for each message that should be sent separately.
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
    return [group for group in groups if group] or [[]]


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
    if not _only_text(chain):
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
    chain: list[BaseMessageComponent],
    nickname: str,
    self_id: str,
) -> list[Nodes]:
    """Wrap plain segments into one forward message sent as the bot.

    Args:
        chain: Text components to forward.
        nickname: Display name used for every node.
        self_id: Bot account id used as the node sender.

    Returns:
        A single ``Nodes`` component.
    """
    content: list[BaseMessageComponent] = [
        comp for comp in chain if isinstance(comp, Plain) and comp.text.strip()
    ]
    node = Node(name=nickname, uin=self_id or "0", content=content or [Plain("")])
    return [Nodes([node])]


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
