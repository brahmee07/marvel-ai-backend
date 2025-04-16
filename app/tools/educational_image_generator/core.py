import os
from app.services.logger import setup_logger
from app.api.error_utilities import ToolExecutorError
from app.utils.document_loaders import get_docs
from langchain_core.documents import Document
from app.tools.image_generator.tools import ImageGenerator

logger = setup_logger()

def load_support_documents(file_url: str, file_type: str, lang: str, verbose: bool) -> list:
    """
    Load support documents from the given file URL and type.
    """
    try:
        logger.info(f"Loading support documents from {file_type} at {file_url}")
        return get_docs(file_url, file_type, lang=lang, verbose=verbose)
    except Exception as e:
        raise ToolExecutorError(f"Failed to load document: {e}")

def build_image_generation_prompt(docs: list, grade_level: str, subject: str, image_style: str,
                                  use_case: str, additional_instructions: str, lang: str, verbose: bool) -> dict:
    """
    Generate the prompt for image generation using ImageGenerator.
    """
    image_gen = ImageGenerator(
        grade_level=grade_level,
        subject=subject,
        image_style=image_style,
        use_case=use_case,
        additional_instructions=additional_instructions,
        lang=lang,
        verbose=verbose
    )
    return image_gen.build_prompt(docs)

def executor(
    prompt: str,
    grade_level: str = "",
    subject: str = "",
    image_style: str = "",
    use_case: str = "",
    additional_instructions: str = "",
    lang: str = "en",
    file_type: str = "",
    file_url: str = "",
    verbose: bool = True
):
    """
    Executor function for the Image Generator tool.

    Args:
        prompt (str): Base prompt or description of the image to generate.
        grade_level (str): Target grade level.
        subject (str): Subject area (e.g., Math, Science).
        image_style (str): Type of visual style (e.g., Cartoon, Diagram).
        use_case (str): Intended use (e.g., worksheet, slide).
        additional_instructions (str): Extra guidance for prompt generation.
        lang (str): Language preference.
        file_type (str): Optional supporting document type (e.g., PDF, DOCX).
        file_url (str): Optional URL to supporting file.
        verbose (bool): Whether to log verbosely.

    Returns:
        dict: A dictionary with the full prompt used for image generation.

    Raises:
        ToolExecutorError: If an error occurs during the execution pipeline.
    """
    try:
        docs = []

        # Load documents if URL and type are provided
        if file_url and file_type:
            docs.extend(load_support_documents(file_url, file_type, lang, verbose))

        # Add the main prompt as a document
        if prompt:
            docs.append(Document(page_content=prompt))

        # Build the image generation prompt
        output = build_image_generation_prompt(docs, grade_level, subject, image_style, use_case, 
                                               additional_instructions, lang, verbose)

        logger.info("Image generation prompt built successfully.")
        return output

    except Exception as e:
        error_message = f"Error in image generator executor: {e}"
        logger.error(error_message)
        raise ToolExecutorError(error_message)
