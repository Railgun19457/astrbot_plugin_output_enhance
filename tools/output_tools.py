"""LLM tools that send content markers cannot express."""

from __future__ import annotations

from typing import Any

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Node, Nodes, Plain
from astrbot.core.agent.tool import FunctionTool
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ..core.config import supports
from ..core.pipeline import image_component
from ..core.renderer import render_text


@pydantic_dataclass
class SendForwardTool(FunctionTool[None]):
    """Send a custom merged-forward message as the bot itself."""

    plugin: Any = Field(default=None, repr=False, exclude=True)
    name: str = "send_forward_message"
    description: str = (
        "发送一条合并转发消息。只在需要把多段内容折叠成聊天记录时使用。"
        "nodes 的每一项包含 nickname 和 content，content 只能是文本。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "转发节点列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nickname": {
                                "type": "string",
                                "description": "节点显示昵称，可省略",
                            },
                            "content": {
                                "type": "string",
                                "description": "节点文本内容",
                            },
                        },
                        "required": ["content"],
                    },
                }
            },
            "required": ["nodes"],
        }
    )

    async def run(self, event: AstrMessageEvent, nodes: list[dict]) -> str:
        """Send the forward message and report the result.

        Args:
            event: Event whose session receives the message.
            nodes: Node nickname and text pairs supplied by the model.

        Returns:
            A short result the model can recall on the next turn.
        """
        plugin = self.plugin
        if plugin is None:
            return "发送失败：工具未就绪。"
        if not supports(event.get_platform_name(), "forward"):
            return "当前平台不支持合并转发，未发送。"
        if not isinstance(nodes, list) or not nodes:
            return "发送失败：没有可转发的节点。"

        built: list[Node] = []
        for item in nodes:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            nickname = (
                str(item.get("nickname") or "").strip()
                or plugin.config.default_nickname
            )
            # QQ resolves a real member uin to that member's group card and
            # discards the custom nickname inside the opened forward. "0" is
            # not a member, so the nickname supplied here is kept.
            built.append(Node(name=nickname, uin="0", content=[Plain(content)]))
        if not built:
            return "发送失败：节点内容为空。"

        await event.send(MessageChain([Nodes(built)]))
        return f"已发送合并转发，包含 {len(built)} 个节点。"


@pydantic_dataclass
class SendTextAsImageTool(FunctionTool[None]):
    """Render supplied text and send it as an image."""

    plugin: Any = Field(default=None, repr=False, exclude=True)
    name: str = "send_text_as_image"
    description: str = (
        "把给定文本渲染成图片并发送。参数 text 只能是要展示的文本，"
        "不要传入图片路径、文件路径或 URL。发送已有图片请使用 send_message_to_user。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要渲染成图片的文本",
                }
            },
            "required": ["text"],
        }
    )

    async def run(self, event: AstrMessageEvent, text: str) -> str:
        """Render text and send the resulting images.

        Args:
            event: Event whose session receives the images.
            text: Text to render. Paths and URLs are rejected.

        Returns:
            A short result the model can recall on the next turn.
        """
        plugin = self.plugin
        if plugin is None:
            return "发送失败：工具未就绪。"
        content = str(text or "").strip()
        if not content:
            return "发送失败：文本为空。"
        if content.startswith(("http://", "https://", "file://")) or _looks_like_path(
            content
        ):
            return "发送失败：这个工具只接收文本，不接收图片路径或 URL。"

        rendered = await render_text(content, plugin.config, plugin.font_path)
        if not rendered:
            return "发送失败：文本渲染没有产生图片。"
        await event.send(MessageChain([image_component(path) for path in rendered]))
        return f"已把文本渲染为 {len(rendered)} 张图片并发送。"


def _looks_like_path(text: str) -> bool:
    stripped = text.strip().strip("\"'")
    if "\n" in stripped or " " in stripped:
        return False
    suffix = stripped.rsplit(".", 1)[-1].lower() if "." in stripped else ""
    return suffix in {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


def build_tools(plugin: Any) -> list[FunctionTool[None]]:
    """Create the tools selected in the plugin config.

    Args:
        plugin: Plugin instance stored on each tool.

    Returns:
        Tools that should be registered with AstrBot.
    """
    selected = plugin.config.llm_tool_options
    tools: list[FunctionTool[None]] = []
    if "send_forward_message" in selected:
        tools.append(SendForwardTool(plugin=plugin))
    if "send_text_as_image" in selected:
        tools.append(SendTextAsImageTool(plugin=plugin))
    return tools
