"""Turn plain current-session tool sends back into normal replies."""

from __future__ import annotations

import inspect
from typing import Any

from astrbot.api import logger

SEND_MESSAGE_TOOL_NAME = "send_message_to_user"
PROACTIVE_PROMPT_MARKER = "You are now responding to a scheduled task."
_NORMAL_CHAT_MARKER = "output_enhance_normal_chat"
_REQUEST_MARKER = "_output_enhance_normal_chat"
_PATCH_MARKER = "_output_enhance_send_tool_patch"

_runner_cls: type | None = None
_runner_method: Any = None
_stage_cls: type | None = None
_stage_method: Any = None
_installed = False
_users = 0


def install() -> bool:
    """Patch the runner and agent stage used by plain tool sends.

    Returns:
        True when both patches are installed. A missing AstrBot entry point
        leaves the previous methods untouched.
    """
    global _installed, _users
    if _installed:
        _users += 1
        return True

    runner_cls = _load_runner_cls()
    stage_cls = _load_stage_cls()
    if runner_cls is None or stage_cls is None:
        return False

    runner_method = runner_cls._iter_llm_responses_with_fallback
    stage_method = stage_cls.process
    if not _is_async_generator(runner_method) or not _is_async_generator(stage_method):
        logger.warning(
            "[OutputEnhance] AstrBot send-tool entry points changed; "
            "plain tool sends will not be converted."
        )
        return False

    async def iter_responses(runner: Any):
        async for response in runner_method(runner):
            yield rewrite_plain_send(runner, response)

    async def process_stage(stage: Any, event: Any, provider_wake_prefix: str):
        mark_normal_chat(event)
        async for item in stage_method(stage, event, provider_wake_prefix):
            yield item

    setattr(iter_responses, _PATCH_MARKER, True)
    setattr(process_stage, _PATCH_MARKER, True)
    runner_cls._iter_llm_responses_with_fallback = iter_responses
    stage_cls.process = process_stage

    global _runner_cls, _runner_method, _stage_cls, _stage_method
    _runner_cls = runner_cls
    _runner_method = runner_method
    _stage_cls = stage_cls
    _stage_method = stage_method
    _installed = True
    _users = 1
    logger.info("[OutputEnhance] Plain send_message_to_user calls will be converted.")
    return True


def uninstall() -> None:
    """Restore the methods captured when the patch was installed."""
    global _installed, _users, _runner_cls, _runner_method, _stage_cls, _stage_method
    if not _installed:
        return
    _users -= 1
    if _users > 0:
        return
    if (
        _runner_cls is not None
        and _runner_method is not None
        and getattr(_runner_cls._iter_llm_responses_with_fallback, _PATCH_MARKER, False)
    ):
        _runner_cls._iter_llm_responses_with_fallback = _runner_method
    if (
        _stage_cls is not None
        and _stage_method is not None
        and getattr(_stage_cls.process, _PATCH_MARKER, False)
    ):
        _stage_cls.process = _stage_method
    _runner_cls = None
    _runner_method = None
    _stage_cls = None
    _stage_method = None
    _installed = False
    _users = 0


def mark_request(event: Any, req: Any) -> None:
    """Remember the main chat request that should stop streaming.

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
    if not isinstance(args, list) or len(args) != 1 or not isinstance(args[0], dict):
        return None
    if not _same_session(args[0].get("session"), _current_session(runner)):
        return None

    messages = args[0].get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        if str(message.get("type", "")).lower() != "plain":
            return None
        text = str(message.get("text", "")).strip()
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
    try:
        return type(response)(
            role="assistant",
            completion_text=text,
            result_chain=None,
            tools_call_args=[],
            tools_call_name=[],
            tools_call_ids=[],
            tools_call_extra_content={},
            reasoning_content=getattr(response, "reasoning_content", None),
            reasoning_signature=getattr(response, "reasoning_signature", None),
            raw_completion=getattr(response, "raw_completion", None),
            is_chunk=False,
            id=getattr(response, "id", None),
            usage=getattr(response, "usage", None),
        )
    except (TypeError, ValueError):
        logger.exception(
            "[OutputEnhance] Could not rebuild the model response; keeping the tool call."
        )
        return response


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


def _load_runner_cls() -> type | None:
    try:
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )
    except ImportError:
        logger.warning(
            "[OutputEnhance] ToolLoopAgentRunner is unavailable; "
            "plain tool sends will not be converted."
        )
        return None
    return ToolLoopAgentRunner


def _load_stage_cls() -> type | None:
    try:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
            InternalAgentSubStage,
        )
    except ImportError:
        logger.warning(
            "[OutputEnhance] InternalAgentSubStage is unavailable; "
            "plain tool sends will not be converted."
        )
        return None
    return InternalAgentSubStage


def _is_async_generator(method: object) -> bool:
    return inspect.isasyncgenfunction(method)
