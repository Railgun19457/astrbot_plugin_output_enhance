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
    background: tuple[int, int, int]
    foreground: tuple[int, int, int]
    muted: tuple[int, int, int]
    code_background: tuple[int, int, int]
    quote: tuple[int, int, int]


_FALLBACK_TEMPLATE = PillowTemplate(
    960,
    64,
    32,
    1.7,
    1800,
    (255, 255, 255),
    (36, 36, 36),
    (107, 114, 128),
    (244, 244, 245),
    (209, 213, 219),
)
# Noto Color Emoji is a bitmap font and only renders at this pixel size.
_EMOJI_BITMAP_SIZE = 109
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
    fonts = _font_set(font_path, config.emoji_font_path, template.font_size)
    blocks = _markdown_blocks(text)
    content_width = template.width - template.margin * 2
    laid_out = [
        _layout_block(block, fonts, template, content_width) for block in blocks
    ]
    capacity = max(template.max_height - template.margin * 2, 1)
    pages: list[list[_LaidBlock]] = [[]]
    used = 0
    for block in laid_out:
        if config.auto_page and pages[-1] and used + block.height > capacity:
            pages.append([])
            used = 0
        pages[-1].append(block)
        used += block.height
    directory = Path(get_astrbot_temp_path()) / "output_enhance"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for page in pages:
        height = template.margin * 2 + max(sum(block.height for block in page), 1)
        image = Image.new("RGBA", (template.width, height), (*template.background, 255))
        draw = ImageDraw.Draw(image)
        y = template.margin
        for block in page:
            _paint_block(draw, block, fonts, template, y)
            y += block.height
        path = directory / f"{uuid.uuid4().hex}.png"
        image.convert("RGB").save(path, "PNG")
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
        for name in ("background", "foreground", "muted", "code_background", "quote")
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
        background=colors["background"],
        foreground=colors["foreground"],
        muted=colors["muted"],
        code_background=colors["code_background"],
        quote=colors["quote"],
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


class _Run(NamedTuple):
    """One uniformly styled piece of a Markdown line."""

    text: str
    bold: bool = False
    italic: bool = False
    strike: bool = False
    code: bool = False


class _Block(NamedTuple):
    """One Markdown block before it is wrapped to the image width."""

    kind: str
    lines: tuple[tuple[_Run, ...], ...]
    indent: int = 0


class _LaidBlock(NamedTuple):
    """One wrapped block and the vertical space it occupies."""

    kind: str
    lines: tuple[tuple[_Run, ...], ...]
    height: int
    indent: int = 0


class _Fonts(NamedTuple):
    """Body fonts plus the fixed-size color emoji font."""

    regular: ImageFont.ImageFont
    bold: ImageFont.ImageFont
    italic: ImageFont.ImageFont
    code: ImageFont.ImageFont
    emoji: ImageFont.ImageFont | None


def _font_set(font_path: str, emoji_path: str, size: int) -> _Fonts:
    """Load the body fonts and the color emoji font.

    Args:
        font_path: Preferred CJK body font.
        emoji_path: Color emoji font. It has one valid bitmap size.
        size: Body font size requested by the template.

    Returns:
        Fonts used while measuring and painting Markdown.
    """
    regular = _load_font(font_path, size)
    bold = _load_font(font_path, size, index=1) or regular
    italic = _load_font(font_path, size, index=2) or regular
    code = _load_font("CascadiaMono.ttf", size, required=False) or _load_font(
        "consola.ttf",
        size,
        required=False,
    )
    emoji = _load_emoji(emoji_path)
    return _Fonts(regular, bold, italic, code or regular, emoji)


def _load_emoji(font_path: str) -> ImageFont.ImageFont | None:
    """Load a bitmap color emoji font at its only supported size."""
    if not font_path:
        return None
    try:
        return ImageFont.truetype(font_path, _EMOJI_BITMAP_SIZE)
    except OSError:
        logger.warning("[OutputEnhance] Color emoji font unavailable: %s", font_path)
        return None


def _markdown_blocks(text: str) -> list[_Block]:
    """Parse the Markdown subset used by local rendering.

    Args:
        text: Source Markdown.

    Returns:
        Headings, paragraphs, lists, quotes, fenced code, and tables. Marker
        characters are consumed instead of being drawn.
    """
    blocks: list[_Block] = []
    paragraph: list[str] = []
    lines = text.splitlines()
    index = 0

    def flush() -> None:
        if paragraph:
            blocks.append(_Block("paragraph", (_inline(" ".join(paragraph)),)))
            paragraph.clear()

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("```"):
            flush()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            blocks.append(
                _Block("code", tuple((_Run(line, code=True),) for line in code))
            )
            index += 1
            continue
        if _table_at(lines, index):
            flush()
            rows = []
            while index < len(lines) and _is_table_row(lines[index]):
                if not _is_table_separator(lines[index]):
                    rows.append(_table_cells(lines[index]))
                index += 1
            blocks.append(_Block("table", tuple(rows)))
            continue
        if not stripped:
            flush()
            index += 1
            continue
        heading = re.match(r"(#{1,3})\s+(.*)", stripped)
        quote = re.match(r">\s?(.*)", stripped)
        bullet = re.match(r"(\s*)[-*+]\s+(.*)", lines[index])
        ordered = re.match(r"(\s*)(\d+)[.)]\s+(.*)", lines[index])
        if heading:
            flush()
            level = len(heading.group(1))
            blocks.append(_Block(f"h{level}", (_inline(heading.group(2)),)))
        elif quote:
            flush()
            blocks.append(_Block("quote", (_inline(quote.group(1)),)))
        elif bullet:
            flush()
            blocks.append(
                _Block("bullet", (_inline(bullet.group(2)),), len(bullet.group(1)))
            )
        elif ordered:
            flush()
            blocks.append(
                _Block(
                    "ordered",
                    (_inline(ordered.group(3)),),
                    len(ordered.group(1)),
                )
            )
        else:
            paragraph.append(stripped)
        index += 1
    flush()
    return blocks or [_Block("paragraph", ((_Run(""),),))]


def _inline(text: str) -> tuple[_Run, ...]:
    """Split inline bold, strike, italic, and code markers into runs."""
    pattern = re.compile(r"(\*\*.+?\*\*|~~.+?~~|`.+?`|\*.+?\*)")
    runs: list[_Run] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            runs.append(_Run(text[cursor : match.start()]))
        token = match.group()
        if token.startswith("**"):
            runs.append(_Run(token[2:-2], bold=True))
        elif token.startswith("~~"):
            runs.append(_Run(token[2:-2], strike=True))
        elif token.startswith("`"):
            runs.append(_Run(token[1:-1], code=True))
        else:
            runs.append(_Run(token[1:-1], italic=True))
        cursor = match.end()
    if cursor < len(text):
        runs.append(_Run(text[cursor:]))
    return tuple(run for run in runs if run.text)


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def _is_table_separator(line: str) -> bool:
    return bool(
        re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line.strip())
    )


def _table_at(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and _is_table_row(lines[index])
        and _is_table_separator(lines[index + 1])
    )


def _table_cells(line: str) -> tuple[_Run, ...]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return tuple(_Run(cell) for cell in cells)


def _layout_block(
    block: _Block,
    fonts: _Fonts,
    template: PillowTemplate,
    width: int,
) -> _LaidBlock:
    """Wrap one Markdown block and measure its rendered height."""
    line_height = max(int(template.font_size * template.line_height), 1)
    if block.kind == "code":
        lines = tuple(
            _wrap_runs(line, fonts, width - template.font_size) for line in block.lines
        )
        flat = tuple(line for group in lines for line in group)
        return _LaidBlock(block.kind, flat, line_height * (len(flat) + 1))
    if block.kind == "table":
        return _LaidBlock(
            block.kind,
            block.lines,
            line_height * (len(block.lines) + 1),
        )
    indent = template.font_size * 2 if block.kind in {"bullet", "ordered"} else 0
    wrapped = _wrap_runs(block.lines[0], fonts, width - indent)
    gap = line_height if block.kind.startswith("h") else line_height // 2
    return _LaidBlock(block.kind, wrapped, line_height * len(wrapped) + gap, indent)


def _paint_block(
    draw: ImageDraw.ImageDraw,
    block: _LaidBlock,
    fonts: _Fonts,
    template: PillowTemplate,
    y: int,
) -> None:
    """Paint one wrapped Markdown block."""
    x = template.margin + block.indent
    line_height = max(int(template.font_size * template.line_height), 1)
    if block.kind == "code":
        draw.rounded_rectangle(
            (
                template.margin,
                y,
                template.width - template.margin,
                y + block.height - line_height // 2,
            ),
            radius=12,
            fill=template.code_background,
        )
    if block.kind == "quote":
        draw.rectangle(
            (template.margin, y, template.margin + 4, y + line_height),
            fill=template.quote,
        )
        x += template.font_size
    for index, line in enumerate(block.lines):
        prefix = ""
        if block.kind == "bullet" and index == 0:
            prefix = "• "
        elif block.kind == "ordered" and index == 0:
            prefix = "1. "
        _paint_runs(
            draw,
            (x, y + index * line_height),
            ((_Run(prefix),) if prefix else ()) + line,
            fonts,
            template,
        )


def _paint_runs(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    runs: tuple[_Run, ...],
    fonts: _Fonts,
    template: PillowTemplate,
) -> None:
    """Paint styled runs and scaled color emoji."""
    x, y = origin
    for run in runs:
        for kind, piece in _runs(run.text):
            if kind == "emoji" and fonts.emoji is not None:
                target = template.font_size
                glyph = Image.new("RGBA", (_EMOJI_BITMAP_SIZE * 2, _EMOJI_BITMAP_SIZE))
                ImageDraw.Draw(glyph).text(
                    (0, 0),
                    piece,
                    font=fonts.emoji,
                    embedded_color=True,
                )
                box = glyph.getbbox()
                if box:
                    glyph = glyph.crop(box)
                    glyph.thumbnail((target, target), Image.Resampling.LANCZOS)
                    image = getattr(draw, "_image", None)
                    if isinstance(image, Image.Image):
                        image.alpha_composite(
                            glyph,
                            (x, y + max((target - glyph.height) // 2, 0)),
                        )
                    x += glyph.width
                continue
            font = _run_font(run, fonts)
            fill = template.muted if run.code else template.foreground
            draw.text((x, y), piece, font=font, fill=fill)
            if run.strike:
                middle = y + template.font_size // 2
                draw.line((x, middle, x + font.getlength(piece), middle), fill=fill)
            x += int(font.getlength(piece))


def _run_font(run: _Run, fonts: _Fonts) -> ImageFont.ImageFont:
    if run.code:
        return fonts.code
    if run.bold:
        return fonts.bold
    if run.italic:
        return fonts.italic
    return fonts.regular


def _wrap_runs(
    runs: tuple[_Run, ...],
    fonts: _Fonts,
    width: int,
) -> tuple[tuple[_Run, ...], ...]:
    """Wrap styled runs without drawing their Markdown markers."""
    lines: list[list[_Run]] = [[]]
    used = 0
    for run in runs:
        pending = run
        while pending.text:
            font = _run_font(pending, fonts)
            taken = ""
            taken_width = 0
            for char in pending.text:
                active = (
                    fonts.emoji if _EMOJI_RE.fullmatch(char) and fonts.emoji else font
                )
                char_width = (
                    int(active.getlength(char)) if active is font else font.size
                )
                if taken and used + taken_width + char_width > width:
                    break
                taken += char
                taken_width += char_width
            lines[-1].append(
                _Run(taken, pending.bold, pending.italic, pending.strike, pending.code)
            )
            used += taken_width
            pending = _Run(
                pending.text[len(taken) :],
                pending.bold,
                pending.italic,
                pending.strike,
                pending.code,
            )
            if pending.text:
                lines.append([])
                used = 0
    return tuple(tuple(line) for line in lines if line)


def _load_font(
    font_path: str,
    size: int,
    *,
    index: int = 0,
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
            return ImageFont.truetype(candidate, size, index=index)
        except OSError:
            continue
    if not required:
        return None
    logger.warning("[OutputEnhance] No CJK font found; text may render as boxes.")
    return ImageFont.load_default()


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
