"""Render reply text to an image and resolve the configured font."""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

from astrbot.api import logger
from astrbot.core import html_renderer
from astrbot.core.utils.io import download_file

from .config import PluginConfig

# A page is split once the rendered text would make the image unreadably tall.
_PAGE_CHAR_LIMIT = 1800


async def resolve_font(config: PluginConfig, data_dir: Path, raw_config: object) -> str:
    """Download a configured font URL and rewrite that config field locally.

    Args:
        config: Normalized config. ``font_path`` is updated in place.
        data_dir: Plugin data directory used to store the downloaded font.
        raw_config: AstrBot config object that supports ``save_config_async``.

    Returns:
        The local font path, or an empty string when no font is configured.
    """
    font_path = config.font_path.strip()
    if not font_path.lower().startswith(("http://", "https://")):
        return font_path

    suffix = Path(urlparse(font_path).path).suffix or ".ttf"
    destination = data_dir / f"font{suffix}"
    try:
        await asyncio.to_thread(data_dir.mkdir, parents=True, exist_ok=True)
        await download_file(font_path, str(destination), show_progress=False)
    except Exception:
        logger.exception("[OutputEnhance] Failed to download font: %s", font_path)
        return ""

    local_path = str(destination)
    config.font_path = local_path
    text_to_image = getattr(raw_config, "get", lambda *_: {})("text_to_image", {})
    if isinstance(text_to_image, dict):
        text_to_image["font_path"] = local_path
    save = getattr(raw_config, "save_config_async", None)
    if save is not None:
        await save()
    logger.info("[OutputEnhance] Font downloaded and config rewritten: %s", local_path)
    return local_path


async def render_text(text: str, config: PluginConfig, font_path: str) -> list[str]:
    """Render text with the framework renderer.

    Args:
        text: Text to render.
        config: Template name and auto-page switch.
        font_path: Local font path. Empty uses the renderer default.

    Returns:
        Local paths or URLs of the rendered images. Long text is split into
        several images when auto paging is enabled.

    Raises:
        Exception: Renderer failures are left to the caller.
    """
    template = config.style_template.strip() or None
    pages = _pages(text) if config.auto_page else [text]
    rendered: list[str] = []
    for page in pages:
        url = await html_renderer.render_t2i(
            page,
            return_url=False,
            use_network=True,
            template_name=template,
        )
        if url:
            rendered.append(str(url))
    if font_path and rendered:
        logger.debug("[OutputEnhance] Custom font is configured: %s", font_path)
    return rendered


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
