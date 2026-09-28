"""Turn plain current-session tool sends back into normal replies."""

from __future__ import annotations

import collections.abc
import functools
import inspect
from typing import Any

from astrbot.api import logger

SEND_MESSAGE_TOOL_NAME = "send_message_to_user"
PROACTIVE_PROMPT_MARKER = "You are now responding to a scheduled task."
_NORMAL_CHAT_MARKER = "output_enhance_normal_chat"
_REQUEST_MARKER = "_output_enhance_normal_chat"
_PATCH_MARKER = "_output_enhance_send_tool_patch"
_PATCHES = (
    (
        "astrbot.core.agent.runners.tool_loop_agent_runner",
        "ToolLoopAgentRunner",
        "_iter_llm_responses_with_fallback",
    ),
    (
        "astrbot.core.agent.runners.tool_loop_agent_runner",
        "ToolLoopAgentRunner",
        "_resolve_tool_exec",
    ),
    (
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal",
        "InternalAgentSubStage",
        "process",
    ),
    (
        "astrbot.core.pipeline.process_stage.method.agent_sub_stages.third_party",
        "ThirdPartyAgentSubStage",
        "process",
    ),
)

_installed = False
_originals: dict[tuple[type, str], Any] = {}


def install() -> bool:
    """Patch the runners and agent stages used by plain tool sends.

    Returns:
        True when every entry point is patched. A missing or unexpected
        AstrBot method leaves previously installed patches untouched.
    """
    global _installed
    if _installed:
        return True

    loaded: list[tuple[type, str, Any]] = []
    for module_name, class_name, method_name in _PATCHES:
        owner = _load_owner(module_name, class_name)
        if owner is None:
            return False
        method = getattr(owner, method_name, None)
        if getattr(method, _PATCH_MARKER, False):
            logger.warning(
                "[OutputEnhance] %s.%s is already patched; "
                "plain tool sends will not be converted.",
                class_name,
                method_name,
            )
            return False
        if not _is_coroutine(method):
            logger.warning(
                "[OutputEnhance] %s.%s changed; "
                "plain tool sends will not be converted.",
                class_name,
                method_name,
            )
            return False
        loaded.append((owner, method_name, method))

    for owner, method_name, method in loaded:
        _originals[(owner, method_name)] = method
        if method_name == "_iter_llm_responses_with_fallback":
            wrapped = _wrap_responses(method)
        elif method_name == "_resolve_tool_exec":
            wrapped = _wrap_requery(method)
        else:
            wrapped = _wrap_stage(method)
        # Keep the captured method reachable after this module is replaced
        # by a plugin reload. Set it after update_wrapper so a copied
        # __dict__ cannot point __wrapped__ at an older layer.
        functools.update_wrapper(wrapped, method)
        wrapped.__wrapped__ = method
        setattr(wrapped, _PATCH_MARKER, True)
        setattr(owner, method_name, wrapped)

    _installed = True
    logger.info("[OutputEnhance] Plain send_message_to_user calls will be converted.")
    return True


def uninstall() -> None:
    """Restore the methods captured when the patch was installed."""
    global _installed
    if not _installed:
        return
    for (owner, method_name), method in list(_originals.items()):
        current = getattr(owner, method_name, None)
        if getattr(current, _PATCH_MARKER, False):
            setattr(owner, method_name, method)
    _originals.clear()
    _installed = False


def _wrap_responses(method: Any):
    async def iter_responses(runner: Any):
        async for response in method(runner):
            yield rewrite_plain_send(runner, response)

    return iter_responses


def _wrap_requery(method: Any):
    async def resolve_tool_exec(runner: Any, response: Any):
        # skills_like re-queries the model. Apply the same plain-send rule to
        # that second response, otherwise the original tool call still runs.
        result = await method(runner, response)
        if not isinstance(result, tuple) or not result:
            return result
        rewritten = rewrite_plain_send(runner, result[0])
        if rewritten is result[0]:
            return result
        return (rewritten, *result[1:])

    return resolve_tool_exec


def _wrap_stage(method: Any):
    async def process_stage(stage: Any, event: Any, provider_wake_prefix: str):
        mark_normal_chat(event)
        async for item in method(stage, event, provider_wake_prefix):
            yield item

    return process_stage


def mark_request(event: Any, req: Any) -> None:
    """Remember a normal chat request and turn streaming off for this turn.

    Streaming must be decided before the model responds. A converted tool send
    becomes a normal reply, and a streaming finish skips the pre-send hook, so
    the whole turn is non-streaming once this feature is active.

    Args:
        event: Event entering the LLM request hook.
        req: Provider request built for that event.
    """
    if not _marked(event):
        return
    try:
        setattr(req, _REQUEST_MARKER, True)
    except (AttributeError, TypeError):
        logger.debug("[OutputEnhance] Could not mark the provider request.")
        return
    set_extra = getattr(event, "set_extra", None)
    if callable(set_extra):
        set_extra("enable_streaming", False)


def rewrite_plain_send(runner: Any, response: Any) -> Any:
    """Replace one plain current-session tool send with assistant text.

    Args:
        runner: Agent runner that produced the response.
        response: One model response, including streaming chunks.

    Returns:
        The original response, or a copied assistant response whose only
        content is the joined plain text.
    """
    text = extract_plain_send(runner, response)
    if text is None:
        return response
    if getattr(runner, "streaming", False):
        runner.streaming = False
    logger.info(
        "[OutputEnhance] Converted a plain send_message_to_user call into a reply."
    )
    return _assistant_response(response, text)


def extract_plain_send(runner: Any, response: Any) -> str | None:
    """Return text only when this one call can safely become a normal reply.

    Args:
        runner: Agent runner carrying the current event and request.
        response: Completed model response to inspect.

    Returns:
        Joined plain-text parts, or None when the call must stay a tool call.
    """
    if getattr(response, "is_chunk", False):
        return None
    if str(getattr(response, "completion_text", "") or "").strip():
        return None
    chain = getattr(getattr(response, "result_chain", None), "chain", None)
    if isinstance(chain, list) and chain:
        return None
    if _skipped_runner(runner):
        return None

    names = getattr(response, "tools_call_name", None)
    args = getattr(response, "tools_call_args", None)
    if not isinstance(names, list) or names != [SEND_MESSAGE_TOOL_NAME]:
        return None
    if not isinstance(args, list) or len(args) != 1:
        return None
    payload = _as_mapping(args[0])
    if payload is None:
        return None
    if not _same_session(payload.get("session"), _current_session(runner)):
        return None

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    texts: list[str] = []
    for message in messages:
        item = _as_mapping(message)
        if item is None:
            return None
        if str(item.get("type", "")).lower() != "plain":
            return None
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        texts.append(text)
    return "\n".join(texts)


def mark_normal_chat(event: Any) -> None:
    """Mark a user chat so its plain tool sends can rejoin the reply path.

    Args:
        event: Event about to enter the internal agent stage.
    """
    if event is None or _proactive(event):
        return
    get_extra = getattr(event, "get_extra", None)
    if not callable(get_extra) or get_extra("provider_request") is not None:
        return
    set_extra = getattr(event, "set_extra", None)
    if not callable(set_extra):
        return
    set_extra(_NORMAL_CHAT_MARKER, True)
    set_extra("enable_streaming", False)


def _assistant_response(response: Any, text: str) -> Any:
    """Rebuild a tool response as assistant text.

    ``LLMResponse`` reads ``completion_text`` from its result chain whenever
    that chain exists, including an empty one. Gemini creates the empty chain
    before the tool call is known, so passing either the old chain or the text
    into the constructor drops the converted reply.

    Args:
        response: Model response being replaced.
        text: Plain text that should become the user-visible reply.

    Returns:
        The rebuilt response, or the original when its text would not survive.
    """
    try:
        rebuilt = type(response)(role="assistant")
        # Dataclass __init__ assigns its own defaults after this returns, so
        # every field must be written once construction has finished.
        if hasattr(rebuilt, "result_chain"):
            rebuilt.result_chain = None
        rebuilt.tools_call_args = []
        rebuilt.tools_call_name = []
        rebuilt.tools_call_ids = []
        rebuilt.tools_call_extra_content = {}
        rebuilt.reasoning_content = getattr(response, "reasoning_content", None)
        rebuilt.reasoning_signature = getattr(response, "reasoning_signature", None)
        rebuilt.raw_completion = getattr(response, "raw_completion", None)
        rebuilt.is_chunk = False
        rebuilt.id = getattr(response, "id", None)
        rebuilt.usage = getattr(response, "usage", None)
        rebuilt.completion_text = text
    except (AttributeError, TypeError, ValueError):
        logger.exception(
            "[OutputEnhance] Could not rebuild the model response; keeping the tool call."
        )
        return response
    if not _keeps_text(rebuilt, text):
        logger.error(
            "[OutputEnhance] Rebuilt model response dropped the reply text; keeping the tool call."
        )
        return response
    return rebuilt


def _keeps_text(response: Any, text: str) -> bool:
    """Return whether the rebuilt response still exposes the converted text.

    An empty result chain is true for ``LLMResponse``. Its text setter then
    inserts a text component into that chain, so the chain must be ignored
    while reading the stored text.

    Args:
        response: Rebuilt model response.
        text: Text the conversion intended to keep.

    Returns:
        True when either the stored text or the chain text matches exactly.
    """
    chain = getattr(response, "result_chain", None)
    if hasattr(response, "result_chain"):
        response.result_chain = None
    stored = str(getattr(response, "completion_text", "") or "")
    if hasattr(response, "result_chain"):
        response.result_chain = chain
    if stored == text:
        return True
    get_plain_text = getattr(chain, "get_plain_text", None)
    return callable(get_plain_text) and str(get_plain_text() or "") == text


def _skipped_runner(runner: Any) -> bool:
    event = _runner_event(runner)
    if event is None or _proactive(event):
        return True
    req = getattr(runner, "req", None)
    if not _marked(event) or getattr(req, _REQUEST_MARKER, False) is not True:
        return True
    prompt = str(getattr(req, "prompt", "") or "")
    return PROACTIVE_PROMPT_MARKER in prompt


def _proactive(event: Any) -> bool:
    get_extra = getattr(event, "get_extra", None)
    action = get_extra("action_type") if callable(get_extra) else None
    return action == "live" or event.__class__.__name__ == "CronMessageEvent"


def _same_session(session: object, current: str | None) -> bool:
    if not current:
        return False
    if session is None:
        return True
    text = str(session).strip()
    if not text:
        return True
    if text == current:
        return True
    if ":" in text:
        return False
    return text == current.rsplit(":", 1)[-1]


def _current_session(runner: Any) -> str | None:
    event = _runner_event(runner)
    session = getattr(event, "unified_msg_origin", None)
    return str(session) if session else None


def _runner_event(runner: Any) -> Any | None:
    run_context = getattr(runner, "run_context", None)
    context = getattr(run_context, "context", None)
    return getattr(context, "event", None)


def _marked(event: Any) -> bool:
    get_extra = getattr(event, "get_extra", None)
    return callable(get_extra) and get_extra(_NORMAL_CHAT_MARKER) is True


def release_stale_patch() -> None:
    """Drop a patch left behind by a replaced plugin module.

    AstrBot re-executes a plugin module on reload, so this module no longer
    remembers an install performed by the previous copy. The classes do.
    """
    if _installed or _originals:
        return
    for module_name, class_name, method_name in _PATCHES:
        owner = _load_owner(module_name, class_name)
        if owner is None:
            continue
        current = getattr(owner, method_name, None)
        restored = current
        while getattr(restored, _PATCH_MARKER, False):
            wrapped = getattr(restored, "__wrapped__", None)
            if wrapped is None or wrapped is restored:
                logger.warning(
                    "[OutputEnhance] Could not restore %s.%s after reload.",
                    class_name,
                    method_name,
                )
                restored = None
                break
            restored = wrapped
        if restored is not None and restored is not current:
            setattr(owner, method_name, restored)


def _load_owner(module_name: str, class_name: str) -> type | None:
    try:
        module = __import__(module_name, fromlist=[class_name])
        owner = getattr(module, class_name)
    except (ImportError, AttributeError):
        logger.warning(
            "[OutputEnhance] %s is unavailable; plain tool sends will not be converted.",
            class_name,
        )
        return None
    if not isinstance(owner, type):
        logger.warning(
            "[OutputEnhance] %s is unavailable; plain tool sends will not be converted.",
            class_name,
        )
        return None
    return owner


def _as_mapping(value: object) -> collections.abc.Mapping | None:
    """Return a read-only mapping, including Gemini protobuf argument maps."""
    if isinstance(value, collections.abc.Mapping):
        return value
    items = getattr(value, "items", None)
    if not callable(items):
        return None
    try:
        return dict(items())
    except (TypeError, ValueError):
        return None


def _is_coroutine(method: object) -> bool:
    return inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)
