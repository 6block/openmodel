"""Multimodal content parsing for the OpenAI-compatible API layer.

Image understanding (VLM) models accept a message `content` that is either a
plain string (text-only, unchanged from M1) or an OpenAI-style array of parts:
    [{"type": "text", "text": "..."},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

This module normalises that array into two things the engine needs:
  * `rendered` messages — image parts replaced with a template-friendly marker
    so the tokenizer's chat template emits the visual placeholder tokens
    ("<image>", "<|vision_start|>…") for the model.
  * `images` — the decoded PIL Images, in the same order, handed to vLLM's
    `multi_modal_data` channel (NOT embedded in the text prompt).

Keeping text and pixels separate mirrors vLLM's own OpenAI server: the template
renders only text/markers, the raw images travel out-of-band. The receipt still
hashes the RENDERED prompt text and the RESPONSE text, exactly as before — pixels
do not enter the token-count billing path, they only feed the model.
"""

import base64
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Cap on decoded image bytes (default 10 MiB). A single VLM image is typically
# < 2 MiB; this bound prevents a malformed/gigantic data-URI from exhausting the
# worker's memory before vLLM even sees it.
MAX_IMAGE_BYTES = int(__import__("os").environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,\s]*)(?:;(?P<opts>[^,]*))?,(?P<data>.*)$")


@dataclass
class ParsedContent:
    """Result of parsing one message's `content` field."""
    # The message content as a string (text-only) or a list of parts suitable for
    # the tokenizer chat template. Image parts become {"type": "image"} markers.
    rendered: str | list
    # Decoded PIL Images, in the same order the image parts appear in `rendered`.
    images: list
    # True when the content contained at least one image part.
    has_image: bool


def _decode_image_bytes(raw: bytes) -> "object":
    """Decode raw image bytes into a PIL Image (imported lazily — PIL is a vLLM
    transitive dep, not a top-level requirement, and mock/dev envs may lack it)."""
    from PIL import Image

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large: {len(raw)} bytes exceeds {MAX_IMAGE_BYTES}")
    img = Image.open(io.BytesIO(raw))
    img.load()  # force decode; closes nothing, but surfaces a corrupt image now
    return img


def load_image_url(url: str) -> "object":
    """Resolve an image `url` to a PIL Image.

    Accepts, in priority order:
      1. `data:` URIs (base64, possibly URL-encoded) — no network, no fs access.
      2. `file://` or bare filesystem paths.
      3. `http(s)://` URLs — fetched via httpx.

    Remote fetching is deliberately explicit: an operator that does not want the
    worker reaching the internet for image URLs can leave it unused; data-URIs
    and local paths cover the gateway/self-hosted case fully.
    """
    if url.startswith("data:"):
        m = _DATA_URI_RE.match(url)
        if not m:
            raise ValueError("malformed data URI")
        data = m.group("data")
        # URL-encoded payloads (base64 with %2B etc.) — decode the escapes first.
        if "%" in data:
            from urllib.parse import unquote
            data = unquote(data)
        try:
            raw = base64.b64decode(data, validate=True)
        except Exception as e:
            raise ValueError(f"invalid base64 image payload: {e}") from e
        return _decode_image_bytes(raw)

    if url.startswith("file://"):
        p = Path(url[len("file://"):])
        return _decode_image_bytes(p.read_bytes())

    # Bare filesystem path (no scheme).
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return _decode_image_bytes(Path(url).read_bytes())

    # http(s)://
    import httpx

    r = httpx.get(url, follow_redirects=True, timeout=30.0)
    r.raise_for_status()
    return _decode_image_bytes(r.content)


def _render_image_part() -> dict:
    """An image marker for the chat template. Qwen2.5-VL's template recognises
    {"type": "image"}; the actual pixels travel via multi_modal_data."""
    return {"type": "image"}


def parse_content(content) -> ParsedContent:
    """Normalise one message's `content` (str | list) into rendered + images.

    A plain string is returned untouched (has_image=False, images=[]) so the
    text-only path is byte-identical to before this module existed.
    """
    if isinstance(content, str):
        return ParsedContent(rendered=content, images=[], has_image=False)

    if not isinstance(content, list):
        # Unrecognised shape — treat as no content rather than crash the request.
        logger.warning("unsupported content shape %r; treating as empty", type(content))
        return ParsedContent(rendered="", images=[], has_image=False)

    rendered = []
    images = []
    has_image = False
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            rendered.append({"type": "text", "text": str(part.get("text", ""))})
        elif ptype in ("image_url", "image"):
            # Accept both the OpenAI spelling (image_url) and the looser "image".
            if ptype == "image_url":
                iu = part.get("image_url") or {}
                url = iu.get("url", "") if isinstance(iu, dict) else str(iu)
            else:
                url = part.get("url", "")
            if not url:
                raise ValueError(f"{ptype} part missing url")
            try:
                img = load_image_url(str(url))
            except Exception as e:
                raise ValueError(f"failed to load image: {e}") from e
            rendered.append(_render_image_part())
            images.append(img)
            has_image = True
        else:
            # Unknown part type — pass through as text of its string form, so a
            # future part type degrades instead of silently vanishing.
            logger.debug("unknown content part type %r; ignoring", ptype)

    if not has_image:
        # An all-text array (e.g. multiple text parts) collapses to the concatenated
        # text for the template — no image channel involved.
        text = "".join(p.get("text", "") for p in rendered if p.get("type") == "text")
        return ParsedContent(rendered=text, images=[], has_image=False)

    return ParsedContent(rendered=rendered, images=images, has_image=True)
