import logging
from typing import Optional
from pydantic import BaseModel
from google.cloud import vertex_ai


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Pydantic Models
class ImageGenerationPrompt(BaseModel):
    prompt: str
    grade_level: str
    subject: str
    image_style: str
    use_case: str
    additional_instructions: str
    lang: str

class ImageGenerationResponse(BaseModel):
    image_url: str
    status: str
    error_message: Optional[str] = None

# Image Generator
class ImageGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the ImageGenerator with the API key.
        """
        self.api_key = api_key
        self.client = vertex_ai.PredictionServiceClient()

    def generate_image(self, prompt_data: ImageGenerationPrompt) -> ImageGenerationResponse:
        """
        Generate an image based on the provided Pydantic prompt data.
        """
        try:
            payload = prompt_data.dict()

            response = self._call_vertex_ai(payload)

            if 'image_url' in response:
                return ImageGenerationResponse(image_url=response['image_url'], status="success")
            else:
                return ImageGenerationResponse(
                    image_url="",
                    status="failure",
                    error_message="No image URL returned."
                )

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return ImageGenerationResponse(
                image_url="",
                status="failure",
                error_message=str(e)
            )

    def _call_vertex_ai(self, payload: dict):
        """
        Makes the API call to Vertex AI's image generation service.
        """
        try:
            endpoint = "text-to-image-v1"  #not so sure about the end point
            request = {
                "endpoint": endpoint,
                "body": payload
            }

            # Adapt to Vertex AI's actual method if different
            response = self.client.predict(
                endpoint=request['endpoint'],
                body=request['body']
            )

            return response  # should be a dict with 'image_url' if successful

        except Exception as e:
            logger.error(f"Error in API call: {e}")
            raise
