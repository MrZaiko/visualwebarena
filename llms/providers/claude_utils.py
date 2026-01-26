"""Tools to generate from Claude/Anthropic prompts."""

import base64
import io
import logging
import os
import random
import time
from datetime import datetime
from typing import Any

import anthropic
from PIL import Image


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                logger = logging.getLogger("logger")
                logger.error(
                    f"Anthropic API error (attempt {num_retries + 1}): {type(e).__name__}"
                )
                logger.error(f"Error message: {e}")

                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded. Last error: {e}"
                    )

                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)

            except Exception as e:
                logger = logging.getLogger("logger")
                logger.error(f"Unexpected error in Anthropic API call: {e}")
                raise e

    return wrapper


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode("utf-8")


# Anthropic has a 5MB limit for images
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB


def _resize_image_if_needed(base64_data: str, media_type: str) -> tuple[str, str]:
    """Resize image if it exceeds Anthropic's 5MB limit.

    Args:
        base64_data: Base64 encoded image data
        media_type: MIME type of the image (e.g., 'image/png')

    Returns:
        Tuple of (resized_base64_data, media_type)
    """
    # Decode base64 to check size
    image_bytes = base64.standard_b64decode(base64_data)

    if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES:
        return base64_data, media_type

    logger = logging.getLogger("logger")
    logger.debug(f"Image size {len(image_bytes)} bytes exceeds 5MB limit, resizing...")

    # Load image with PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (for JPEG output)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Calculate resize ratio to get under 5MB
    # Start with current size and progressively reduce
    quality = 85
    scale = 1.0

    while True:
        # Resize image
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            resized = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            resized = image

        # Save to buffer as JPEG (better compression than PNG)
        buffered = io.BytesIO()
        resized.save(buffered, format="JPEG", quality=quality, optimize=True)

        if buffered.tell() <= MAX_IMAGE_SIZE_BYTES:
            buffered.seek(0)
            new_base64 = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
            logger.debug(
                f"Resized image to {buffered.tell()} bytes (scale={scale}, quality={quality})"
            )
            return new_base64, "image/jpeg"

        # Reduce quality first, then scale
        if quality > 50:
            quality -= 10
        else:
            scale *= 0.8
            quality = 85  # Reset quality when scaling down

        # Safety check to prevent infinite loop
        if scale < 0.1:
            logger.warning("Could not resize image to under 5MB, using minimum size")
            buffered.seek(0)
            new_base64 = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
            return new_base64, "image/jpeg"


@retry_with_exponential_backoff
def generate_from_claude_chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Generate response from Claude chat completion API.

    Args:
        messages: List of message dictionaries in OpenAI-like format.
                  Will be converted to Anthropic format.
        model: The Claude model to use (e.g., "claude-sonnet-4-20250514").
        temperature: Temperature for sampling.
        max_tokens: Maximum tokens to generate.
        top_p: Top-p sampling parameter.
        context_length: Context length (unused but kept for API compatibility).
        stop_token: Optional stop token (unused but kept for API compatibility).

    Returns:
        Generated text response.
    """
    logger = logging.getLogger("logger")
    logger.debug("Generating from Claude chat completion")

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable must be set when using Anthropic API."
        )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Convert OpenAI-style messages to Anthropic format
    system_message = None
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        name = msg.get("name", "")

        if role == "system":
            # Handle system messages - Anthropic uses a separate system parameter
            if name in ("example_user", "example_assistant"):
                # These are few-shot examples, convert to user/assistant messages
                if name == "example_user":
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": _convert_content_to_anthropic(content),
                        }
                    )
                else:  # example_assistant
                    anthropic_messages.append(
                        {
                            "role": "assistant",
                            "content": _convert_content_to_anthropic(content),
                        }
                    )
            else:
                # Main system message
                if isinstance(content, list):
                    # Extract text from content list
                    system_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            system_parts.append(item)
                    system_message = "\n".join(system_parts)
                else:
                    system_message = content
        elif role == "user":
            anthropic_messages.append(
                {"role": "user", "content": _convert_content_to_anthropic(content)}
            )
        elif role == "assistant":
            anthropic_messages.append(
                {"role": "assistant", "content": _convert_content_to_anthropic(content)}
            )

    # Ensure messages alternate between user and assistant
    anthropic_messages = _ensure_alternating_roles(anthropic_messages)

    # Add cache_control to every message content block
    anthropic_messages = _add_cache_control(anthropic_messages)

    # Build the API call parameters
    api_params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": anthropic_messages,
    }

    if system_message:
        api_params["system"] = [
            {
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if "LOG_FOLDER" in os.environ:
        timestamp_uid = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # e.g., 20260122_104512
        log_path = f"{os.environ['LOG_FOLDER']}/anthropic_{timestamp_uid}.json"
        try:
            import json

            with open(log_path, "x") as f:
                f.write(json.dumps({"messages": messages}) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log to {log_path}: {e}")

    response = client.messages.create(**api_params)

    logger.debug("Received response from Claude chat completion")
    logger.debug(f"Response: {response}")

    # Extract text from response
    answer = ""
    for block in response.content:
        if block.type == "text":
            answer += block.text

    return answer


def _convert_content_to_anthropic(content: Any) -> list[dict[str, Any]] | str:
    """Convert OpenAI-style content to Anthropic format.

    Handles both text-only content and multimodal content with images.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        anthropic_content = []
        for item in content:
            if isinstance(item, str):
                anthropic_content.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    anthropic_content.append(
                        {"type": "text", "text": item.get("text", "")}
                    )
                elif item_type == "image_url":
                    # Convert OpenAI image_url format to Anthropic format
                    image_url = item.get("image_url", {})
                    url = (
                        image_url.get("url", "")
                        if isinstance(image_url, dict)
                        else image_url
                    )

                    if url.startswith("data:"):
                        # Parse data URL: data:image/png;base64,<data>
                        # Format: data:<media_type>;base64,<base64_data>
                        header, base64_data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]

                        # Resize image if needed to stay under Anthropic's 5MB limit
                        base64_data, media_type = _resize_image_if_needed(
                            base64_data, media_type
                        )

                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                },
                            }
                        )
                    else:
                        # URL-based image - Anthropic also supports URLs
                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                },
                            }
                        )
        return anthropic_content

    return str(content)


def _add_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add cache_control breakpoints to maximize cached prefix length.

    Places at most 3 cache_control breakpoints in messages (the system message
    gets its own, totalling up to 4). Breakpoints are placed on the last 3
    messages to maximize the prefix that gets cached, since Anthropic caches
    everything up to and including the block with cache_control.
    """
    if not messages:
        return messages

    # Pick the indices of up to 3 messages to place cache breakpoints on.
    # Strategy: evenly space them across the message list, always including the
    # last message. This maximizes prefix reuse across multi-turn conversations.
    n = len(messages)
    if n <= 3:
        breakpoint_indices = set(range(n))
    else:
        # Place breakpoints at roughly evenly-spaced positions, always
        # including the last message. This gives us good cache coverage:
        # one early (covers few-shot examples), one mid, one at the end.
        breakpoint_indices = set()
        for i in range(3):
            idx = int(i * (n - 1) / 2)
            breakpoint_indices.add(idx)
        # Always include the last message
        breakpoint_indices.add(n - 1)
        # If we have more than 3 due to rounding, keep the last 3
        if len(breakpoint_indices) > 3:
            breakpoint_indices = set(sorted(breakpoint_indices)[-3:])

    for i, msg in enumerate(messages):
        if i not in breakpoint_indices:
            # Ensure content is in list form but without cache_control
            content = msg["content"]
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            continue

        content = msg["content"]
        if isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}

    return messages


def _ensure_alternating_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure messages alternate between user and assistant roles.

    Anthropic requires that messages alternate between user and assistant.
    This function merges consecutive messages of the same role.
    """
    if not messages:
        return messages

    result = []
    for msg in messages:
        if not result:
            result.append(msg)
        elif result[-1]["role"] == msg["role"]:
            # Merge with previous message of the same role
            prev_content = result[-1]["content"]
            curr_content = msg["content"]

            if isinstance(prev_content, str) and isinstance(curr_content, str):
                result[-1]["content"] = prev_content + "\n\n" + curr_content
            elif isinstance(prev_content, list) and isinstance(curr_content, list):
                result[-1]["content"] = prev_content + curr_content
            elif isinstance(prev_content, str) and isinstance(curr_content, list):
                result[-1]["content"] = [
                    {"type": "text", "text": prev_content}
                ] + curr_content
            elif isinstance(prev_content, list) and isinstance(curr_content, str):
                result[-1]["content"] = prev_content + [
                    {"type": "text", "text": curr_content}
                ]
        else:
            result.append(msg)

    return result


@retry_with_exponential_backoff
def fake_generate_from_claude_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Debug only - returns a fake response."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable must be set when using Anthropic API."
        )

    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. In summary, the next action I will perform is ```click [60]```"
    return answer
