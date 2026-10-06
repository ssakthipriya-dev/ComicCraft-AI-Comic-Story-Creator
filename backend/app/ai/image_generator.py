import os
import io
import base64
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw, ImageFont

from .gemini_client import gemini_client
from ..config import settings
from ..utils.logger import get_logger


logger = get_logger("image_generator")


class ImageProvider(ABC):
    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        panel_number: int,
        title: str = "",
        art_style: str = "Comic Book",
    ) -> str:
        pass


class GeminiImageProvider(ImageProvider):
    """Generate comic panel artwork using Gemini image generation."""

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        panel_number: int,
        title: str = "",
        art_style: str = "Comic Book",
    ) -> str:

        if not gemini_client.is_configured:
            raise ValueError("Gemini API key is not configured.")

        try:
            logger.info(
                f"Generating Gemini image for panel {panel_number} "
                f"using {gemini_client.image_model}"
            )

            full_prompt = f"""
Create a professional comic-book panel.

Panel number: {panel_number}
Panel title: {title}
Art style: {art_style}

Scene description:
{prompt}

Requirements:
- High-quality comic illustration
- Cinematic composition
- Detailed environment
- Expressive characters
- Strong visual storytelling
- Consistent character appearance
- 4:3 landscape composition
- No speech bubbles
- No captions
- No watermark
- No unnecessary text
"""

            interaction = gemini_client._client.interactions.create(
                model=gemini_client.image_model,
                input=full_prompt,
                response_format={
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "aspect_ratio": "4:3",
                    "image_size": "1K",
                },
            )

            if not interaction.output_image:
                raise RuntimeError("Gemini returned no image.")

            # Gemini returns JPEG image data
            image_bytes = base64.b64decode(
                interaction.output_image.data
            )

            output_dir = os.path.dirname(output_path)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Convert Gemini JPEG to PNG because the application
            # expects PNG panel files.
            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            image.save(
                output_path,
                format="PNG",
            )

            logger.info(
                f"Gemini image saved successfully: {output_path}"
            )

            return output_path

        except Exception as e:
            logger.exception(
                f"Gemini image generation failed for panel "
                f"{panel_number}: {e}"
            )

            logger.warning(
                "Falling back to local comic artwork generator."
            )

            fallback = FallbackImageProvider()

            return fallback.generate_image(
                prompt,
                output_path,
                panel_number,
                title,
                art_style,
            )


class FallbackImageProvider(ImageProvider):
    """Generate a local placeholder comic panel if Gemini is unavailable."""

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        panel_number: int,
        title: str = "",
        art_style: str = "Comic Book",
    ) -> str:

        width = 800
        height = 600

        image = Image.new(
            "RGB",
            (width, height),
            (235, 235, 235),
        )

        draw = ImageDraw.Draw(image)

        # Background
        draw.rectangle(
            [0, 0, width, height],
            fill=(225, 225, 225),
        )

        # Sky
        draw.rectangle(
            [0, 0, width, 360],
            fill=(175, 205, 230),
        )

        # Ground
        draw.rectangle(
            [0, 360, width, height],
            fill=(95, 125, 80),
        )

        # Sun
        draw.ellipse(
            [620, 70, 730, 180],
            fill=(255, 210, 80),
        )

        # Mountains
        draw.polygon(
            [
                (0, 360),
                (150, 210),
                (300, 360),
            ],
            fill=(100, 110, 125),
        )

        draw.polygon(
            [
                (180, 360),
                (390, 180),
                (610, 360),
            ],
            fill=(80, 95, 110),
        )

        draw.polygon(
            [
                (450, 360),
                (650, 220),
                (800, 360),
            ],
            fill=(110, 120, 135),
        )

        # Character silhouette
        center_x = 400

        draw.ellipse(
            [
                center_x - 45,
                250,
                center_x + 45,
                340,
            ],
            fill=(35, 35, 45),
        )

        draw.rectangle(
            [
                center_x - 60,
                330,
                center_x + 60,
                470,
            ],
            fill=(35, 35, 45),
        )

        draw.line(
            [
                center_x - 60,
                350,
                center_x - 130,
                420,
            ],
            fill=(35, 35, 45),
            width=18,
        )

        draw.line(
            [
                center_x + 60,
                350,
                center_x + 130,
                420,
            ],
            fill=(35, 35, 45),
            width=18,
        )

        # Comic border
        draw.rectangle(
            [5, 5, width - 5, height - 5],
            outline=(20, 20, 20),
            width=10,
        )

        # Font
        try:
            font = ImageFont.truetype(
                "arial.ttf",
                28,
            )
        except Exception:
            font = ImageFont.load_default()

        # Panel number
        draw.text(
            (25, 20),
            f"Panel {panel_number}",
            fill=(20, 20, 20),
            font=font,
        )

        # Fallback notice
        draw.text(
            (25, 530),
            "Fallback Engine Active",
            fill=(20, 20, 20),
            font=font,
        )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        image.save(
            output_path,
            format="PNG",
        )

        logger.info(
            f"Fallback image saved successfully: {output_path}"
        )

        return output_path


def get_image_provider() -> ImageProvider:
    provider_name = settings.IMAGE_PROVIDER.lower()

    if (
        provider_name == "gemini"
        and gemini_client.is_configured
    ):
        return GeminiImageProvider()

    return FallbackImageProvider()