"""Enhance bot output with markers, passive formatting, and send tools."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.star.star_tools import StarTools

from .core.config import PluginConfig, load_config
from .core.pipeline import (
    image_component,
    prepare_chain,
    should_forward,
    should_render_image,
    to_nodes,
)
from .core.renderer import render_text, resolve_font
from .core.send_tool import bind_text_sender, mark_request, release_stale_patch
from .core.send_tool import install as install_send_tool
from .core.send_tool import uninstall as uninstall_send_tool
from .core.text_ops import typing_delay
from .tools.output_tools import build_tools

PLUGIN_NAME = "astrbot_plugin_output_enhance"

# AstrBot settings that duplicate a plugin feature. The plugin only warns.
_CONFLICTS = (
    (
        "segmented_reply",
        ("platform_settings", "segmented_reply", "enable"),
        "platform_settings.segmented_reply.enable",
    ),
    (
        "text_to_image",
        ("t2i",),
        "t2i",
    ),
    (
        "markers",
        ("provider_settings", "streaming_response"),
        "provider_settings.streaming_response",
    ),
    (
        "quote_reply",
        ("platform_settings", "reply_with_quote"),
        "platform_settings.reply_with_quote",
    ),
)


class OutputEnhancePlugin(Star):
    """Parse output markers and apply formatting before a reply is sent."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.raw_config = config if config is not None else {}
        self.config: PluginConfig = load_config(
            self.raw_config
            if isinstance(self.raw_config, dict)
            else dict(self.raw_config)
        )
        self.data_dir: Path = StarTools.get_data_dir(PLUGIN_NAME)
        self.font_path = self.config.font_path
        self._registered_tools: list[str] = []
        self._send_tool_installed = False
        # A reloaded module cannot see the previous install. Restore those
        # methods before deciding whether this copy should patch them again.
        release_stale_patch()
        if self.config.plain_tool_send:
            bind_text_sender(self._send_extracted_text)
            self._send_tool_installed = install_send_tool()
        for tool in build_tools(self):
            self.context.add_llm_tools(tool)
            self._registered_tools.append(tool.name)
        self._warn_conflicts()
        logger.info(
            "[OutputEnhance] Loaded. Tools: %s",
            ", ".join(self._registered_tools) or "none",
        )

    async def initialize(self) -> None:
        """Download a remote font once and keep the local path."""
        if self.config.font_path.lower().startswith(("http://", "https://")):
            self.font_path = await resolve_font(
                self.config, self.data_dir, self.raw_config
            )

    async def terminate(self) -> None:
        """Drop tools registered by this plugin."""
        if self._send_tool_installed:
            bind_text_sender(None)
            uninstall_send_tool()
            self._send_tool_installed = False
        for name in self._registered_tools:
            self.context.unregister_llm_tool(name)
        self._registered_tools.clear()

    @filter.on_llm_request()
    async def inject_marker_prompt(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Append the static marker instructions to the system prompt."""
        if self.config.plain_tool_send:
            mark_request(event, req)
        prompt = self.config.injection_text()
        if not prompt:
            return
        current = req.system_prompt or ""
        req.system_prompt = f"{current}\n\n{prompt}" if current else prompt

    @filter.on_decorating_result(priority=-100)
    async def decorate_output(self, event: AstrMessageEvent) -> None:
        """Parse markers and apply passive output rules before sending."""
        result = event.get_result()
        if result is None or not result.chain:
            return
        if result.result_content_type in {
            ResultContentType.STREAMING_RESULT,
            ResultContentType.STREAMING_FINISH,
        }:
            return

        original = list(result.chain)
        try:
            groups, force_image = prepare_chain(event, original, self.config)
        except Exception:
            logger.exception("[OutputEnhance] Failed to prepare the reply.")
            return
        chain = [comp for group in groups for comp in group]
        if not chain:
            event.clear_result()
            return

        plain_text = "".join(comp.text for comp in chain if isinstance(comp, Plain))
        if self.config.block_enable and any(
            keyword and keyword in plain_text for keyword in self.config.block_keywords
        ):
            logger.info("[OutputEnhance] Reply dropped by message block.")
            event.clear_result()
            return
        if self.config.error_enable and any(
            keyword and keyword in plain_text for keyword in self.config.error_keywords
        ):
            await self._intercept(event, plain_text)
            return

        try:
            if should_render_image(chain, self.config, force_image):
                rendered = await render_text(plain_text, self.config, self.font_path)
                if rendered:
                    result.chain = [image_component(path) for path in rendered]
                    result.use_t2i(False)
                    return
            if should_forward(chain, self.config, event.get_platform_name()):
                segments = groups if len(groups) > 1 else [chain]
                result.chain = to_nodes(segments, self.config.default_nickname)
                return
        except Exception:
            logger.exception(
                "[OutputEnhance] Passive formatting failed; sending the original text."
            )
            result.chain = original
            return

        if len(groups) > 1:
            asyncio.get_running_loop().create_task(self._send_segments(event, groups))
            event.clear_result()
            return
        result.chain = chain

    @filter.on_plugin_error()
    async def intercept_plugin_error(
        self,
        event: AstrMessageEvent,
        plugin_name: str,
        handler_name: str,
        error: Exception,
        traceback_text: str,
    ) -> None:
        """Replace AstrBot's automatic plugin-error echo."""
        if not self.config.error_enable or not self.config.plugin_error:
            return
        detail = (
            f"插件 {plugin_name} 的处理函数 {handler_name} 出现异常：{error}\n\n"
            f"{traceback_text}"
        )
        await self._intercept(event, detail)

    async def _intercept(self, event: AstrMessageEvent, detail: str) -> None:
        """Drop the original reply, optionally notice the user, and forward it.

        The event itself is not stopped. Decoration runs while the agent stage
        is waiting at a yield; stopping here would skip ``astr_agent_complete``
        and the conversation-history write.
        """
        logger.info("[OutputEnhance] Reply intercepted.")
        event.clear_result()
        notice = self.config.user_notice.strip()
        if notice:
            await event.send(MessageChain([Plain(notice)]))
        for session in self.config.forward_sessions:
            try:
                await self.context.send_message(session, MessageChain([Plain(detail)]))
            except Exception:
                logger.exception(
                    "[OutputEnhance] Failed to forward the intercepted error to %s.",
                    session,
                )

    async def _send_extracted_text(self, event: AstrMessageEvent, text: str) -> bool:
        """Format one extracted tool-text run and send it.

        Args:
            event: Event whose session receives the text.
            text: Plain text removed from a mixed tool call.

        Returns:
            True when the processed text is handed to the sender.
        """
        try:
            groups, _ = prepare_chain(
                event,
                [Plain(text)],
                replace(self.config, seg_plugin_messages=True),
            )
        except Exception:  # noqa: BLE001
            logger.exception("[OutputEnhance] Failed to prepare extracted tool text.")
            return False
        groups = [group for group in groups if group]
        if not groups:
            return True
        try:
            if len(groups) > 1:
                await self._send_segments(event, groups)
            else:
                await event.send(MessageChain(groups[0]))
        except Exception:  # noqa: BLE001
            logger.exception("[OutputEnhance] Failed to send extracted tool text.")
            return False
        return True

    async def _send_segments(self, event: AstrMessageEvent, groups: list[list]) -> None:
        """Send each segment after a typing delay.

        Args:
            event: Event whose session receives the segments.
            groups: Component groups produced by explicit or automatic splits.
        """
        for group in groups:
            text = "".join(comp.text for comp in group if isinstance(comp, Plain))
            await asyncio.sleep(typing_delay(text, self.config.seg_typing_speed))
            if group:
                await event.send(MessageChain(group))

    def _warn_conflicts(self) -> None:
        """Warn when an AstrBot setting would fight this plugin."""
        astrbot_config = self.context.get_config()
        enabled = {
            "segmented_reply": self.config.seg_enable,
            "text_to_image": self.config.t2i_enable,
            "markers": any(
                (
                    self.config.at_enable,
                    self.config.quote_enable,
                    self.config.seg_enable,
                    self.config.t2i_enable,
                )
            ),
            "quote_reply": (
                self.config.quote_enable and self.config.auto_quote_interval > 0
            ),
        }
        conflicts = [
            setting
            for feature, path, setting in _CONFLICTS
            if enabled[feature] and _config_enabled(astrbot_config, path)
        ]
        if conflicts:
            logger.warning(
                "[OutputEnhance] AstrBot settings still enabled and may run before "
                "or duplicate this plugin: %s. Turn them off in WebUI.",
                ", ".join(conflicts),
            )


def _config_enabled(config: object, path: tuple[str, ...]) -> bool:
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return bool(current)
