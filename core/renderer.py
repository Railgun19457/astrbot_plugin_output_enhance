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
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from pygments.util import ClassNotFound

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
        for name in (
            "background",
            "foreground",
            "muted",
            "code_background",
            "quote",
        )
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
    color: str = ""


class _Block(NamedTuple):
    """One Markdown block before it is wrapped to the image width."""

    kind: str
    lines: tuple[tuple[_Run, ...], ...]
    indent: int = 0
    language: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    marker: str = ""


class _LaidBlock(NamedTuple):
    """One wrapped block and the vertical space it occupies."""

    kind: str
    lines: tuple[tuple[_Run, ...], ...]
    height: int
    indent: int = 0
    language: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    marker: str = ""


class _Fonts(NamedTuple):
    """Body fonts plus the fixed-size color emoji font."""

    regular: ImageFont.ImageFont
    bold: ImageFont.ImageFont
    italic: ImageFont.ImageFont
    code: ImageFont.ImageFont
    emoji: ImageFont.ImageFont | None
    heading: dict[str, ImageFont.ImageFont]


def _font_set(font_path: str, emoji_path: str, size: int) -> _Fonts:
    """Load the body, heading, emphasis, and color emoji fonts.

    Args:
        font_path: Preferred CJK body font.
        emoji_path: Color emoji font. It has one valid bitmap size.
        size: Body font size requested by the template.

    Returns:
        Fonts used while measuring and painting Markdown.
    """
    regular = _load_font(font_path, size)
    bold = _styled_font(font_path, size, "Bold")
    italic = _styled_font(font_path, size, "Italic")
    code = _monospace_font(int(size * 0.9))
    heading = {
        "h1": _styled_font(font_path, int(size * 1.65), "Bold"),
        "h2": _styled_font(font_path, int(size * 1.35), "Bold"),
        "h3": _styled_font(font_path, int(size * 1.15), "Bold"),
    }
    return _Fonts(
        regular, bold, italic, code or regular, _load_emoji(emoji_path), heading
    )


def _monospace_font(size: int) -> ImageFont.ImageFont | None:
    """Load the first installed monospace face.

    Args:
        size: Code font size.

    Returns:
        A monospace font, or None when none of the known faces exist.
    """
    for name in ("CascadiaMono.ttf", "consola.ttf", "UbuntuMono-Regular.ttf"):
        loaded = _load_font(name, size, required=False)
        if loaded is not None and loaded.getlength("i") == loaded.getlength("M"):
            return loaded
    return None


def _styled_font(font_path: str, size: int, style: str) -> ImageFont.ImageFont:
    """Load a named font style, then synthesize it from the regular font.

    Args:
        font_path: Preferred font path or family name.
        size: Requested pixel size.
        style: ``Bold`` or ``Italic``.

    Returns:
        The styled font. A missing style falls back to the regular font.
    """
    family = Path(font_path).stem or font_path
    for candidate in (f"{family}-{style}", f"{family}{style}", font_path):
        loaded = _load_font(candidate, size, required=False)
        if loaded is not None:
            return loaded
    return _load_font(font_path, size)


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
            language = stripped[3:].strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            blocks.append(
                _Block(
                    "code",
                    tuple((_Run(line, code=True),) for line in code),
                    language=language,
                )
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
            blocks.append(_Block("table", tuple(rows), rows=tuple(rows)))
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
                    marker=f"{ordered.group(2)}. ",
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


def _table_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _layout_block(
    block: _Block,
    fonts: _Fonts,
    template: PillowTemplate,
    width: int,
) -> _LaidBlock:
    """Wrap one Markdown block and measure its rendered height.

    Args:
        block: Parsed Markdown block.
        fonts: Fonts used to measure wrapped lines.
        template: Spacing and font size.
        width: Content width available inside the page margins.

    Returns:
        The wrapped block and the vertical space it occupies.
    """
    base = fonts.heading.get(block.kind, fonts.regular)
    step = _text_height(base, template)
    if block.kind == "code":
        lines = tuple(
            _wrap_runs(line, fonts, width - template.font_size, fonts.code)
            for line in block.lines
        )
        flat = tuple(line for group in lines for line in group) or ((_Run(""),),)
        return _LaidBlock(
            block.kind,
            flat,
            step * (len(flat) + 1),
            language=block.language,
        )
    if block.kind == "table":
        columns = max((len(row) for row in block.rows), default=1)
        return _LaidBlock(
            block.kind,
            (),
            step * (len(block.rows) + 1),
            rows=tuple(row + ("",) * (columns - len(row)) for row in block.rows),
        )
    marker = "• " if block.kind == "bullet" else block.marker
    prefix = int(base.getlength(marker)) + 12 if marker else 0
    quote = template.font_size if block.kind == "quote" else 0
    wrapped = _wrap_runs(block.lines[0], fonts, max(width - prefix - quote, 1), base)
    gap = step // 2 if block.kind.startswith("h") else step // 3
    return _LaidBlock(
        block.kind,
        wrapped,
        step * len(wrapped) + gap,
        prefix + quote,
        marker=marker,
    )


def _paint_block(
    draw: ImageDraw.ImageDraw,
    block: _LaidBlock,
    fonts: _Fonts,
    template: PillowTemplate,
    y: int,
) -> None:
    """Paint one wrapped Markdown block.

    Args:
        draw: Pillow drawing context of the page.
        block: Wrapped block produced by ``_layout_block``.
        fonts: Body, emphasis, and emoji fonts.
        template: Colors and margins.
        y: Top of the block.
    """
    base = fonts.heading.get(block.kind, fonts.regular)
    step = _text_height(base, template)
    pad = max(template.font_size // 3, 8)
    if block.kind == "code":
        draw.rounded_rectangle(
            (
                template.margin,
                y,
                template.width - template.margin,
                y + block.height - step // 3,
            ),
            radius=14,
            fill=template.code_background,
        )
        line_y = y + pad // 2
        for line in _highlight(block.lines, block.language, template):
            _paint_runs(
                draw,
                (template.margin + pad, line_y),
                line,
                fonts,
                template,
                fonts.code,
            )
            line_y += step
        return
    if block.kind == "table":
        _paint_table(draw, block, fonts, template, y, step)
        return
    x = template.margin + block.indent
    if block.kind == "quote":
        draw.rounded_rectangle(
            (
                template.margin,
                y,
                template.width - template.margin,
                y + block.height - step // 3,
            ),
            radius=12,
            fill=template.code_background,
        )
        draw.rectangle(
            (template.margin, y, template.margin + 6, y + block.height - step // 3),
            fill=template.muted,
        )
    for index, line in enumerate(block.lines):
        prefix = block.marker if index == 0 else ""
        _paint_runs(
            draw,
            (x - (int(base.getlength(prefix)) + 12 if prefix else 0), y + index * step),
            ((_Run(prefix, bold=True),) if prefix else ()) + line,
            fonts,
            template,
            base,
        )


def _highlight(
    lines: tuple[tuple[_Run, ...], ...],
    language: str,
    template: PillowTemplate,
) -> tuple[tuple[_Run, ...], ...]:
    """Color one fenced code block when its language is known.

    Args:
        lines: Wrapped source lines.
        language: Fence language. An unknown name stays uncolored.
        template: Page colors. A dark page uses lighter syntax colors.

    Returns:
        One run sequence per source line.
    """
    source = "\n".join("".join(run.text for run in line) for line in lines)
    if not language:
        return tuple((_Run(line, code=True),) for line in source.split("\n"))
    dark = sum(template.background) < 384
    colors = {
        Token.Keyword: "#C4B5FD" if dark else "#7C3AED",
        Token.Name.Function: "#7DD3FC" if dark else "#0369A1",
        Token.Name.Class: "#FDBA74" if dark else "#C2410C",
        Token.String: "#86EFAC" if dark else "#047857",
        Token.Number: "#FDBA74" if dark else "#C2410C",
        Token.Comment: "#9CA3AF" if dark else "#6B7280",
        Token.Operator: "#FDA4AF" if dark else "#BE123C",
        Token.Punctuation: "#D1D5DB" if dark else "#4B5563",
    }
    try:
        tokens = get_lexer_by_name(language).get_tokens(source)
    except ClassNotFound:
        return tuple((_Run(line, code=True),) for line in source.split("\n"))
    painted: list[list[_Run]] = [[]]
    for kind, text in tokens:
        color = ""
        probe = kind
        while probe is not Token:
            color = colors.get(probe, color)
            probe = probe.parent
        parts = text.split("\n")
        for index, part in enumerate(parts):
            if part:
                painted[-1].append(_Run(part, code=True, color=color))
            if index < len(parts) - 1:
                painted.append([])
    return tuple(tuple(line) or (_Run("", code=True),) for line in painted)


def _paint_table(
    draw: ImageDraw.ImageDraw,
    block: _LaidBlock,
    fonts: _Fonts,
    template: PillowTemplate,
    y: int,
    step: int,
) -> None:
    """Paint a bordered table with a shaded header row.

    Args:
        draw: Pillow drawing context of the page.
        block: Table block. ``rows`` holds the cell text.
        fonts: Body font used inside cells.
        template: Border and header colors.
        y: Top of the table.
        step: One row height.
    """
    columns = max((len(row) for row in block.rows), default=1)
    left = template.margin
    width = template.width - template.margin * 2
    column_width = width // columns
    bottom = y + step * len(block.rows)
    for index, row in enumerate(block.rows):
        top = y + index * step
        if index == 0:
            draw.rectangle(
                (left, top, left + width, top + step), fill=template.code_background
            )
        for column, cell in enumerate(row):
            cell_left = left + column * column_width
            draw.rectangle(
                (cell_left, top, cell_left + column_width, top + step),
                outline=template.quote,
                width=1,
            )
            font = fonts.bold if index == 0 else fonts.regular
            box = font.getbbox("国")
            _paint_runs(
                draw,
                (cell_left + 16, top + (step - (box[3] - box[1])) // 2 - box[1]),
                (_Run(cell, bold=index == 0),),
                fonts,
                template,
                font,
                column_width - 32,
            )
    draw.rectangle((left, y, left + width, bottom), outline=template.quote, width=2)


def _paint_runs(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    runs: tuple[_Run, ...],
    fonts: _Fonts,
    template: PillowTemplate,
    base_font: ImageFont.ImageFont | None = None,
    max_width: int | None = None,
) -> None:
    """Paint styled runs, inline-code chips, and scaled color emoji.

    Args:
        draw: Pillow drawing context of the page.
        origin: Top-left of the first run.
        runs: Styled pieces. Newlines start a new visual line.
        fonts: Body, emphasis, code, and emoji fonts.
        template: Colors and font size.
        base_font: Font used by plain runs. Headings pass their larger font.
        max_width: Optional pixel limit, used to keep table cells inside borders.
    """
    x, y = origin
    base = base_font or fonts.regular
    body_box = fonts.regular.getbbox("国")
    emoji_size = max(body_box[3] - body_box[1], 1)
    line_box = base.getbbox("M" if base is fonts.code else "国")
    line_center = (line_box[1] + line_box[3]) // 2
    for run in runs:
        for kind, piece in _runs(run.text):
            if kind == "emoji" and fonts.emoji is not None:
                canvas = _EMOJI_BITMAP_SIZE + 32
                glyph = Image.new("RGBA", (canvas, canvas))
                ImageDraw.Draw(glyph).text(
                    (16, 16),
                    piece,
                    font=fonts.emoji,
                    embedded_color=True,
                )
                box = glyph.getbbox()
                if box:
                    glyph = glyph.crop(box)
                    fitted = Image.new("RGBA", (emoji_size, emoji_size))
                    scale = min(
                        emoji_size / glyph.width,
                        emoji_size / glyph.height,
                        1,
                    )
                    resized = glyph.resize(
                        (
                            max(int(glyph.width * scale), 1),
                            max(int(glyph.height * scale), 1),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    fitted.alpha_composite(
                        resized,
                        (
                            (emoji_size - resized.width) // 2,
                            emoji_size - resized.height,
                        ),
                    )
                    image = getattr(draw, "_image", None)
                    if isinstance(image, Image.Image):
                        image.alpha_composite(
                            fitted,
                            (x, y + line_center - emoji_size // 2),
                        )
                    x += emoji_size + 4
                continue
            font = _run_font(run, fonts, base)
            fill = (
                _color(run.color, template.foreground)
                if run.color
                else template.foreground
            )
            fallback = fonts.regular if base is fonts.code else base
            if run.code and _needs_body_font(font, fallback, piece):
                groups = _font_groups(piece, font, fallback)
            else:
                groups = ((font, piece),)
            drawn = sum(int(item_font.getlength(item)) for item_font, item in groups)
            if max_width is not None and x + drawn > origin[0] + max_width:
                break
            inline = run.code and base is not fonts.code
            if inline:
                pad_x = max(template.font_size // 5, 6)
                pad_y = max(template.font_size // 12, 2)
                draw.rounded_rectangle(
                    (
                        x - pad_x,
                        y + body_box[1] - pad_y,
                        x + drawn + pad_x,
                        y + body_box[3] + pad_y,
                    ),
                    radius=8,
                    fill=template.code_background,
                )
            for item_font, item in groups:
                item_fill = fill
                item_box = item_font.getbbox(
                    "国" if item_font is not fonts.code else "M"
                )
                item_y = y + line_center - (item_box[1] + item_box[3]) // 2
                if run.italic and not run.code:
                    _paint_oblique(
                        draw, (x, item_y), item, item_font, item_fill, run.bold
                    )
                else:
                    draw.text(
                        (x, item_y),
                        item,
                        font=item_font,
                        fill=item_fill,
                        stroke_width=1 if run.bold else 0,
                        stroke_fill=item_fill,
                    )
                if run.strike:
                    box = item_font.getbbox(item or " ")
                    middle = item_y + (box[1] + box[3]) // 2
                    item_width = int(item_font.getlength(item))
                    draw.line(
                        (x, middle, x + item_width, middle),
                        fill=item_fill,
                        width=max(template.font_size // 16, 2),
                    )
                x += int(item_font.getlength(item))
            x += max(template.font_size // 5, 6) * 2 + 8 if inline else 0


def _paint_oblique(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    bold: bool,
) -> None:
    """Skew a missing italic face so CJK emphasis stays visible.

    Args:
        draw: Pillow drawing context of the page.
        origin: Top-left of the unskewed text.
        text: Characters to draw.
        font: Regular or bold font already selected for the run.
        fill: Text color.
        bold: Whether the skewed text also gets a one-pixel stroke.
    """
    image = getattr(draw, "_image", None)
    if not isinstance(image, Image.Image):
        return
    size = max(getattr(font, "size", 32), 1)
    cursor = 0
    for char in text:
        glyph_width = max(int(font.getlength(char)), 1)
        layer = Image.new("RGBA", (glyph_width + size, size * 2))
        ImageDraw.Draw(layer).text(
            (size // 4, 0),
            char,
            font=font,
            fill=(*fill, 255),
            stroke_width=1 if bold else 0,
            stroke_fill=(*fill, 255),
        )
        skewed = layer.transform(
            layer.size,
            Image.Transform.AFFINE,
            (1, 0.42, -size // 5, 0, 1, 0),
            Image.Resampling.BICUBIC,
        )
        image.alpha_composite(skewed, (origin[0] + cursor - size // 4, origin[1]))
        cursor += glyph_width


def _run_font(
    run: _Run,
    fonts: _Fonts,
    base_font: ImageFont.ImageFont | None = None,
) -> ImageFont.ImageFont:
    """Select the face for one run.

    Args:
        run: Styled piece of text.
        fonts: Loaded body, code, and emphasis fonts.
        base_font: Font used when the run has no extra emphasis.

    Returns:
        The font Pillow should measure and draw.
    """
    if run.code:
        return fonts.code
    if run.bold and base_font in {None, fonts.regular}:
        return fonts.bold
    return base_font or fonts.regular


def _text_height(font: ImageFont.ImageFont, template: PillowTemplate) -> int:
    """Return the line step for a font, including the template leading.

    Args:
        font: Font that will draw the line.
        template: Line-height multiplier.

    Returns:
        Pixels from one baseline row to the next.
    """
    box = font.getbbox("国Ag")
    return max(int((box[3] - box[1]) * template.line_height), 1)


def _wrap_runs(
    runs: tuple[_Run, ...],
    fonts: _Fonts,
    width: int,
    base_font: ImageFont.ImageFont | None = None,
) -> tuple[tuple[_Run, ...], ...]:
    """Wrap styled runs without drawing their Markdown markers.

    Args:
        runs: Inline pieces of one source line.
        fonts: Fonts used to measure each piece.
        width: Maximum line width in pixels.
        base_font: Font used by plain runs.

    Returns:
        Wrapped lines. Empty input becomes one empty line.
    """
    lines: list[list[_Run]] = [[]]
    used = 0
    for run in runs:
        pending = run
        while pending.text:
            font = _run_font(pending, fonts, base_font)
            taken = ""
            taken_width = 0
            inline = pending.code and base_font is not fonts.code
            pad = max(getattr(font, "size", 32) // 6, 4) * 2 if inline else 0
            for char in pending.text:
                emoji = bool(_EMOJI_RE.fullmatch(char) and fonts.emoji)
                char_width = (
                    getattr(font, "size", 32) if emoji else int(font.getlength(char))
                )
                if taken and not inline and used + taken_width + char_width > width:
                    break
                taken += char
                taken_width += char_width
            if inline and lines[-1] and used + taken_width + pad > width:
                lines.append([])
                used = 0
            lines[-1].append(pending._replace(text=taken))
            used += taken_width
            pending = pending._replace(text=pending.text[len(taken) :])
            if pending.text:
                lines.append([])
                used = 0
    return tuple(tuple(line) for line in lines if line) or ((_Run(""),),)


def _load_font(
    font_path: str,
    size: int,
    *,
    index: int = 0,
    required: bool = True,
) -> ImageFont.ImageFont | None:
    """Load a font file, falling back to a CJK system font when required.

    Args:
        font_path: Local font path or family name.
        size: Font size requested by the template.
        index: Face index inside a font collection.
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


def _needs_body_font(
    font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    text: str,
) -> bool:
    """Return whether the code font draws a character narrower than body text.

    Args:
        font: Monospace font selected for the code run.
        body_font: CJK font that can draw the missing characters.
        text: Characters about to be drawn.

    Returns:
        True when a non-space character is only a fraction of the body width.
        Cascadia draws its missing-glyph box at the monospace advance, so the
        mask alone cannot identify it.
    """
    return any(
        char.strip() and font.getlength(char) < body_font.getlength(char) * 0.8
        for char in text
    )


def _font_groups(
    text: str,
    code_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> tuple[tuple[ImageFont.ImageFont, str], ...]:
    """Keep monospace characters together and isolate missing glyphs."""
    groups: list[tuple[ImageFont.ImageFont, str]] = []
    for char in text:
        selected = (
            body_font
            if char.strip()
            and code_font.getlength(char) < body_font.getlength(char) * 0.8
            else code_font
        )
        if groups and groups[-1][0] is selected:
            groups[-1] = (selected, groups[-1][1] + char)
        else:
            groups.append((selected, char))
    return tuple(groups)


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
    return runs or [("text", text)]


def _pages(text: str) -> list[str]:
    """Split long framework-renderer text near paragraph breaks.

    Args:
        text: Source text rendered by AstrBot T2I.

    Returns:
        Pages no longer than the character limit.
    """
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
