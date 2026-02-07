"""Image generation tool -- generate images via LiteLLM."""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Any

from nibot.log import logger
from nibot.registry import Tool

_MAX_IMAGES = 4


class ImageGenerationTool(Tool):
    """Generate images from text prompts using DALL-E, Gemini Imagen, etc."""

    def __init__(self, workspace: Path, default_model: str = "") -> None:
        self._workspace = workspace
        self._images_dir = workspace / "images"
        self._default_model = default_model or "dall-e-3"

    @property
    def name(self) -> str:
        return "image_gen"

    @property
    def description(self) -> str:
        return "Generate images from text prompts. Returns saved file paths."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed image description",
                },
                "model": {
                    "type": "string",
                    "description": "Model name (e.g. dall-e-3, gemini/imagen-4.0-generate-001)",
                },
                "size": {
                    "type": "string",
                    "description": "Image dimensions (default: 1024x1024)",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of images (1-4, default: 1)",
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, **kwargs: Any) -> str:
        kwargs.pop("_tool_ctx", None)
        prompt = kwargs.get("prompt", "")
        if not prompt:
            return "Error: prompt is required."
        model = kwargs.get("model", "") or self._default_model
        size = kwargs.get("size", "1024x1024")
        n = min(max(kwargs.get("n", 1), 1), _MAX_IMAGES)

        try:
            from litellm import aimage_generation
        except ImportError:
            return "Error: litellm is not installed or does not support image generation."

        self._images_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = await aimage_generation(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                response_format="b64_json",
                timeout=300,
            )
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return f"Error: image generation failed: {e}"

        saved: list[str] = []
        for item in response.data:
            b64 = getattr(item, "b64_json", None)
            if not b64:
                continue
            # Strip data URI prefix if present
            if "," in b64 and b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64)
            except Exception:
                continue
            filename = f"{uuid.uuid4().hex[:12]}.png"
            path = self._images_dir / filename
            path.write_bytes(img_bytes)
            saved.append(str(path))
            logger.info(f"Image saved: {path}")

        if not saved:
            return "Error: no images were generated."
        return json.dumps({"images": saved, "count": len(saved), "model": model})
