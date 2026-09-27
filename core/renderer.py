"""Render reply text to an image and resolve the configured font."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from astrbot.api import logger
from astrbot.core import html_renderer
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.io import download_file
from PIL import Image, ImageDraw, ImageFont

from .config import PluginConfig

# A page is split once the rendered text would make the image unreadably tall.
_PAGE_CHAR_LIMIT = 1800
_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
_DEFAULT_TEMPLATE = _TEMPLATE_DIR / "light.json"


class PillowTemplate(NamedTuple):
    """Layout and colors loaded from one Pillow template file."""

    width: int
    margin: int
    font_size: int
    line_height: float
    max_height: int
    accent_width: int
    background: tuple[int, int, int]
    foreground: tuple[int, int, int]
    accent: tuple[int, int, int]


_FALLBACK_TEMPLATE = PillowTemplate(
    920,
    56,
    32,
    1.6,
    1800,
    8,
    (247, 246, 243),
    (34, 34, 34),
    (184, 132, 88),
)
_FONT_CANDIDATES = (
    "msyh.ttc",
    "msyh.ttf",
    "PingFang.ttc",
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKsc-Regular.otf",
    "SourceHanSansSC-Regular.otf",
    "simhei.ttf",
    "simsun.ttc",
)
_EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001faff"
    "\U0001f600-\U0001f64f"
    "\u2600-\u27bf"
    "\ufe0e\ufe0f"
    "]+"
)


async def resolve_font(config: PluginConfig, data_dir: Path, raw_config: object) -> str:
    """Download configured font URLs and rewrite those fields locally.

    Args:
        config: Normalized config. Downloaded paths are updated in place.
        data_dir: Plugin data directory used to store the downloaded fonts.
        raw_config: AstrBot config object that supports ``save_config_async``.

    Returns:
        The local body-font path, or an empty string when it cannot be loaded.
    """
    text_to_image = getattr(raw_config, "get", lambda *_: {})("text_to_image", {})
    changed = False
    body = await _download_font(config.font_path, data_dir, "font")
    if body != config.font_path:
        config.font_path = body
        if isinstance(text_to_image, dict):
            text_to_image["font_path"] = body
        changed = True
    emoji = await _download_font(config.emoji_font_path, data_dir, "emoji-font")
    if emoji != config.emoji_font_path:
        config.emoji_font_path = emoji
        if isinstance(text_to_image, dict):
            text_to_image["emoji_font_path"] = emoji
        changed = True
    save = getattr(raw_config, "save_config_async", None)
    if changed and save is not None:
        await save()
        logger.info("[OutputEnhance] Font config rewritten to local paths.")
    return body


async def _download_font(source: str, data_dir: Path, name: str) -> str:
    """Download one font URL into the plugin data directory.

    Args:
        source: Local path or HTTP(S) URL.
        data_dir: Directory that receives the file.
        name: Stable filename stem.

    Returns:
        The local path after a successful download. The original source is
        returned when it is already local or the download fails.
    """
    font_path = source.strip()
    if not font_path.lower().startswith(("http://", "https://")):
        return font_path
    suffix = Path(urlparse(font_path).path).suffix or ".ttf"
    destination = data_dir / f"{name}{suffix}"
    try:
        await asyncio.to_thread(data_dir.mkdir, parents=True, exist_ok=True)
        await download_file(font_path, str(destination), show_progress=False)
    except Exception:
        logger.exception("[OutputEnhance] Failed to download font: %s", font_path)
        return font_path
    return str(destination)


async def render_text(text: str, config: PluginConfig, font_path: str) -> list[str]:
    """Render text with the renderer selected in the plugin config.

    Args:
        text: Text to render.
        config: Renderer choice and the auto-page switch.
        font_path: Local body font used by Pillow.

    Returns:
        Local paths or URLs of the rendered images. AstrBot T2I uses the
        template selected in AstrBot's own text-to-image settings.

    Raises:
        Exception: Renderer failures are left to the caller.
    """
    if config.t2i_renderer == "astrbot_t2i":
        return await _render_framework(text, config)
    return await asyncio.to_thread(_render_pillow, text, config, font_path)


async def _render_framework(text: str, config: PluginConfig) -> list[str]:
    """Render text through AstrBot's HTML renderer.

    Args:
        text: Text to render, including Markdown and formulas.
        config: Auto-page switch. The template comes from AstrBot itself.

    Returns:
        Renderer paths or URLs. Long text is split by characters because the
        HTML renderer does not expose a page height.
    """
    pages = _pages(text) if config.auto_page else [text]
    rendered: list[str] = []
    for page in pages:
        url = await html_renderer.render_t2i(
            page,
            return_url=False,
            use_network=True,
        )
        if url:
            rendered.append(str(url))
    return rendered


def _render_pillow(text: str, config: PluginConfig, font_path: str) -> list[str]:
    """Draw wrapped text into one or more PNG files.

    Args:
        text: Plain text. Markdown and formulas are drawn as characters.
        config: Auto-page switch.
        font_path: Preferred font. A system CJK font is used when it fails.

    Returns:
        Paths of the saved PNG files.
    """
    template = _load_template(config.pillow_template)
    font = _load_font(font_path, template.font_size)
    emoji_font = _load_font(
        config.emoji_font_path,
        template.font_size,
        required=False,
    )
    line_height = max(int(template.font_size * template.line_height), 1)
    content_width = template.width - template.margin * 2
    lines = _wrap(text, font, emoji_font, content_width)
    capacity = max((template.max_height - template.margin * 2) // line_height, 1)
    chunks = (
        [lines[index : index + capacity] for index in range(0, len(lines), capacity)]
        if config.auto_page
        else [lines]
    )
    directory = Path(get_astrbot_temp_path()) / "output_enhance"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for chunk in chunks or [[""]]:
        height = template.margin * 2 + max(len(chunk), 1) * line_height
        image = Image.new("RGB", (template.width, height), template.background)
        draw = ImageDraw.Draw(image)
        if template.accent_width:
            draw.rectangle((0, 0, template.accent_width, height), fill=template.accent)
        for index, line in enumerate(chunk):
            if line:
                _draw_line(
                    draw,
                    (template.margin, template.margin + index * line_height),
                    line,
                    font,
                    emoji_font,
                    template.foreground,
                )
        path = directory / f"{uuid.uuid4().hex}.png"
        image.save(path, "PNG")
        paths.append(str(path))
    return paths


def _load_template(path: str) -> PillowTemplate:
    """Load a Pillow template, falling back to the bundled light template.

    Args:
        path: Empty for the bundled light template, a path relative to the
            plugin directory, or an absolute JSON path.

    Returns:
        A complete template. Missing files and invalid fields use the built-in
        light values.
    """
    selected = Path(path) if path else _DEFAULT_TEMPLATE
    if not selected.is_absolute():
        selected = _TEMPLATE_DIR.parent / selected
    try:
        raw = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("[OutputEnhance] Pillow template unavailable: %s", selected)
        return _FALLBACK_TEMPLATE
    if not isinstance(raw, dict):
        return _FALLBACK_TEMPLATE
    colors = {
        name: _color(raw.get(name), getattr(_FALLBACK_TEMPLATE, name))
        for name in ("background", "foreground", "accent")
    }
    return PillowTemplate(
        width=_positive(raw.get("width"), _FALLBACK_TEMPLATE.width),
        margin=_nonnegative(raw.get("margin"), _FALLBACK_TEMPLATE.margin),
        font_size=_positive(raw.get("font_size"), _FALLBACK_TEMPLATE.font_size),
        line_height=_positive_float(
            raw.get("line_height"),
            _FALLBACK_TEMPLATE.line_height,
        ),
        max_height=_positive(raw.get("max_height"), _FALLBACK_TEMPLATE.max_height),
        accent_width=_nonnegative(
            raw.get("accent_width"),
            _FALLBACK_TEMPLATE.accent_width,
        ),
        background=colors["background"],
        foreground=colors["foreground"],
        accent=colors["accent"],
    )


def _positive(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and value > 0 else fallback


def _nonnegative(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and value >= 0 else fallback


def _positive_float(value: object, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return float(value) if value > 0 else fallback


def _color(value: object, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    """Parse a ``#RRGGBB`` color."""
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return fallback
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def _load_font(
    font_path: str,
    size: int,
    *,
    required: bool = True,
) -> ImageFont.ImageFont | None:
    """Load a font file, falling back to a CJK system font when required.

    Args:
        font_path: Local font path.
        size: Font size requested by the template.
        required: Whether a system font and Pillow's default may be used.

    Returns:
        The loaded font, or None when an optional font cannot be loaded.
    """
    candidates = [font_path, *_FONT_CANDIDATES] if required else [font_path]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    if not required:
        return None
    logger.warning("[OutputEnhance] No CJK font found; text may render as boxes.")
    return ImageFont.load_default()


def _draw_line(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    line: str,
    font: ImageFont.ImageFont,
    emoji_font: ImageFont.ImageFont | None,
    fill: tuple[int, int, int],
) -> None:
    """Draw one line, switching to the emoji font for emoji runs.

    Args:
        draw: Pillow drawing context.
        origin: Top-left position of the line.
        line: Text to draw.
        font: Font used for ordinary text.
        emoji_font: Font used for emoji runs.
        fill: Text color from the selected template.
    """
    x, y = origin
    for kind, piece in _runs(line):
        active = emoji_font if kind == "emoji" and emoji_font is not None else font
        draw.text((x, y), piece, font=active, fill=fill)
        x += int(active.getlength(piece))


def _runs(text: str) -> list[tuple[str, str]]:
    """Split text into body and emoji runs."""
    runs: list[tuple[str, str]] = []
    cursor = 0
    for match in _EMOJI_RE.finditer(text):
        if match.start() > cursor:
            runs.append(("text", text[cursor : match.start()]))
        runs.append(("emoji", match.group()))
        cursor = match.end()
    if cursor < len(text):
        runs.append(("text", text[cursor:]))
    return runs


def _wrap(
    text: str,
    font: ImageFont.ImageFont,
    emoji_font: ImageFont.ImageFont | None,
    width: int,
) -> list[str]:
    """Wrap each paragraph to the image content width.

    Args:
        text: Text to wrap.
        font: Font used for ordinary text.
        emoji_font: Font used for emoji runs. Missing falls back to ``font``.
        width: Maximum line width in pixels.

    Returns:
        Display lines. Empty paragraphs stay empty.
    """
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        current_width = 0
        for char in paragraph:
            active = (
                emoji_font
                if emoji_font is not None and _EMOJI_RE.fullmatch(char)
                else font
            )
            char_width = int(active.getlength(char))
            if current and current_width + char_width > width:
                lines.append(current)
                current = char
                current_width = char_width
            else:
                current += char
                current_width += char_width
        lines.append(current)
    return lines or [""]


def _pages(text: str) -> list[str]:
    if len(text) <= _PAGE_CHAR_LIMIT:
        return [text]
    pages: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= _PAGE_CHAR_LIMIT:
            pages.append(remaining)
            break
        cut = remaining.rfind("\n", 0, _PAGE_CHAR_LIMIT)
        if cut < _PAGE_CHAR_LIMIT // 2:
            cut = _PAGE_CHAR_LIMIT
        pages.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [page for page in pages if page]
