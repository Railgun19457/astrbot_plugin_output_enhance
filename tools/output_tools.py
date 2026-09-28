"""LLM tools that send content markers cannot express."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import (
    At,
    AtAll,
    BaseMessageComponent,
    Face,
    Image,
    Node,
    Nodes,
    Plain,
)
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.utils.astrbot_path import (
    get_astrbot_system_tmp_path,
    get_astrbot_temp_path,
)
from astrbot.core.utils.media_utils import file_uri_to_path
from astrbot.core.workspace import (
    default_workspace_root,
    resolve_workspace_root_for_umo,
)
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
        "每个节点包含 nickname 和 content。content 按显示顺序排列，"
        "可以混排 text、image、face 和 at。"
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
                                "type": "array",
                                "description": "节点内容，按顺序排列",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["text", "image", "face", "at"],
                                            "description": "片段类型",
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "type 为 text 时的文本",
                                        },
                                        "url": {
                                            "type": "string",
                                            "description": (
                                                "image 的地址：http、https，或当前"
                                                "工作区与临时目录内的本地路径"
                                            ),
                                        },
                                        "id": {
                                            "type": "string",
                                            "description": (
                                                "face 的表情 ID，或 at 的用户 ID；"
                                                "全体填 all"
                                            ),
                                        },
                                    },
                                    "required": ["type"],
                                },
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
            nodes: Node nickname and ordered content segments.

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
        allowed_roots = await _image_roots(event.unified_msg_origin)
        for item in nodes:
            if not isinstance(item, dict):
                continue
            content = _node_content(
                item.get("content"),
                allow_at_all=plugin.config.allow_at_all,
                allowed_roots=allowed_roots,
            )
            if not content:
                continue
            nickname = (
                str(item.get("nickname") or "").strip()
                or plugin.config.default_nickname
            )
            # QQ resolves a real member uin to that member's group card and
            # discards the custom nickname inside the opened forward. "0" is
            # not a member, so the nickname supplied here is kept.
            built.append(Node(name=nickname, uin="0", content=content))
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


def _node_content(
    raw: object,
    *,
    allow_at_all: bool,
    allowed_roots: tuple[Path, ...],
) -> list[BaseMessageComponent]:
    """Turn model-supplied node content into message components.

    Args:
        raw: A plain string, or an ordered list of text, image, face, and at
            segments. A plain string is kept for callers that still send text.
        allow_at_all: Whether an ``at`` segment may target everyone.
        allowed_roots: Directories a local image may come from.

    Returns:
        Components in the original order. Unsupported segments are skipped.
    """
    if isinstance(raw, str):
        text = raw.strip()
        return [Plain(text)] if text else []
    if not isinstance(raw, list):
        return []

    parts: list[BaseMessageComponent] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(Plain(text))
            continue
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "text").strip().lower()
        if kind == "text":
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(Plain(text))
        elif kind == "image":
            image = _image_component(
                str(item.get("url") or "").strip(),
                allowed_roots,
            )
            if image is not None:
                parts.append(image)
        elif kind == "face":
            face_id = str(item.get("id") or "").strip()
            if face_id.isdigit():
                parts.append(Face(id=int(face_id)))
        elif kind == "at":
            target = str(item.get("id") or "").strip()
            if target == "all":
                if allow_at_all:
                    parts.append(AtAll())
            elif target.isdigit():
                parts.append(At(qq=target))
    return parts


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


async def _image_roots(origin: str) -> tuple[Path, ...]:
    """Return the only directories a local forward image may use.

    Args:
        origin: Unified message origin of the current session.

    Returns:
        The session workspace, AstrBot temp directory, and shared tool temp
        directory. A missing workspace falls back to the temp directories.
    """
    roots = [
        Path(get_astrbot_temp_path()).resolve(strict=False),
        Path(get_astrbot_system_tmp_path()).resolve(strict=False),
    ]
    if origin:
        try:
            workspace = await resolve_workspace_root_for_umo(origin)
        except Exception:  # noqa: BLE001
            workspace = default_workspace_root(origin)
        roots.insert(0, workspace.resolve(strict=False))
    return tuple(roots)


def _image_component(source: str, allowed_roots: tuple[Path, ...]) -> Image | None:
    """Build an image from a URL or an allowed local file.

    Args:
        source: HTTP(S) URL, ``file://`` URI, or local path.
        allowed_roots: Workspace and temporary directories that may be read.

    Returns:
        An image component, or None when the source is missing, not an image,
        or outside the allowed directories.
    """
    if source.startswith(("http://", "https://")):
        return Image.fromURL(source)
    candidate = Path(file_uri_to_path(source) if source.startswith("file:") else source)
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        if not allowed_roots:
            return None
        candidate = allowed_roots[0] / candidate
    path = candidate.resolve(strict=False)
    allowed = any(path == root or path.is_relative_to(root) for root in allowed_roots)
    if allowed and path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file():
        return Image.fromFileSystem(path)
    return None


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
