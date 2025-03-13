import os
import socket
import json
import asyncio
import logging
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial
import anthropic
from anthropic import RateLimitError
from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult, ToolVersion
from typing import Any, Dict, Optional, cast, get_args
import httpx
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
HTTP Server for Claude Computer Use Demo

Features:
- Supports Claude 3.5 and 3.7 models
- Supports both Anthropic API and AWS Bedrock (Bedrock is default)
- Thinking capability is enabled by default for Claude 3.7 models
- Configurable via environment variables or API
- Streaming responses with Server-Sent Events
- Persistent chat state across requests
"""

# Define validate_aws_credentials function before it's used
def validate_aws_credentials():
    """Validate AWS credentials for Bedrock access."""
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        
        # Check for credentials
        if not credentials:
            logger.warning("No AWS credentials found - please configure AWS credentials")
            logger.info("You can set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            logger.info("Or configure ~/.aws/credentials file")
            return False
        
        # Check credential expiration if temporary
        if hasattr(credentials, 'expiry_time') and credentials.expiry_time:
            expiry = credentials.expiry_time
            now = datetime.now()
            if expiry < now:
                logger.warning(f"AWS credentials have expired on {expiry}")
                return False
            elif expiry < now + timedelta(hours=1):
                logger.warning(f"AWS credentials will expire soon: {expiry}")
        
        # Check for region
        region = os.environ.get("AWS_REGION")
        if not region:
            region = session.region_name
            
        if not region:
            logger.warning("AWS_REGION environment variable not set and no default region found")
            logger.info("Please set AWS_REGION environment variable or configure region in AWS config")
            return False
            
        # Check if we can access Bedrock service
        try:
            # Just initialize a client to check if we can connect to Bedrock
            # We don't actually call any methods
            bedrock_client = session.client('bedrock-runtime', region_name=region)
            logger.info(f"Successfully configured AWS Bedrock with region: {region}")
            return True
        except Exception as e:
            if "could not be found" in str(e):
                logger.error(f"Bedrock service not available in region {region}")
                logger.info("Make sure to use a region where Bedrock is available, like us-east-1 or us-west-2")
            else:
                logger.error(f"Error connecting to Bedrock service: {str(e)}")
            return False
            
    except ImportError:
        logger.error("boto3 package not installed - required for Bedrock access")
        logger.info("Install boto3 with: pip install boto3")
        return False
    except Exception as e:
        logger.error(f"Error validating AWS credentials: {str(e)}")
        return False

# Default configuration
CONFIG_DIR = Path("~/.anthropic").expanduser()
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Constants
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"

@dataclass(kw_only=True, frozen=True)
class ModelConfig:
    tool_version: str
    max_output_tokens: int
    default_output_tokens: int
    has_thinking: bool = False

SONNET_3_5_NEW = ModelConfig(
    tool_version="computer_use_20241022",
    max_output_tokens=1024 * 8,
    default_output_tokens=1024 * 4,
)

SONNET_3_7 = ModelConfig(
    tool_version="computer_use_20250124",
    max_output_tokens=128_000,
    default_output_tokens=1024 * 16,
    has_thinking=True,
)

MODEL_TO_MODEL_CONF: dict[str, ModelConfig] = {
    "claude-3-7-sonnet-20250219": SONNET_3_7,
}

# Get API key from environment variable or config file
def get_api_key() -> str:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        api_key_file = CONFIG_DIR / "api_key"
        if api_key_file.exists():
            api_key = api_key_file.read_text().strip()
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable or config file is not set")
    return api_key

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}")

def format_model_name_for_provider(model: str, provider: str) -> str:
    """Format the model name appropriately for the given provider."""
    if provider == "bedrock":
        # Handle the us.anthropic prefix format
        if model.startswith("us.anthropic."):
            return model
            
        # Ensure Bedrock model names have the 'anthropic.' prefix if not already present
        if not model.startswith("anthropic."):
            # Map Anthropic model names to Bedrock model names
            model_mapping = {
                "claude-3-5-sonnet-20240620": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                "claude-3-7-sonnet-20250219": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            }
            return model_mapping.get(model, DEFAULT_BEDROCK_MODEL)
        return model
    else:
        # For Anthropic API, remove the prefixes if present
        if model.startswith("us.anthropic."):
            parts = model.split(":")
            model_name = parts[0].replace("us.anthropic.", "")
            return model_name
        elif model.startswith("anthropic."):
            parts = model.split(":")
            model_name = parts[0].replace("anthropic.", "")
            return model_name
        return model

class ServerConfig:
    def __init__(self):
        # Load configuration from environment or config files
        self.api_provider = os.getenv("API_PROVIDER", "bedrock")
        model_env = os.getenv("MODEL")
        if model_env:
            self.model = model_env
        else:
            self.model = DEFAULT_BEDROCK_MODEL if self.api_provider == "bedrock" else DEFAULT_MODEL
        
        # Format model name appropriately for the provider
        self.model = format_model_name_for_provider(self.model, self.api_provider)
        
        # Only get API key for Anthropic provider
        self.api_key = get_api_key() if self.api_provider == "anthropic" else None
        self.custom_system_prompt = load_from_storage("system_prompt") or ""
        self.only_n_most_recent_images = int(os.getenv("ONLY_N_MOST_RECENT_IMAGES", "90"))
        self.hide_images = os.getenv("HIDE_IMAGES", "false").lower() == "true"
        self.token_efficient_tools_beta = os.getenv("TOKEN_EFFICIENT_TOOLS_BETA", "false").lower() == "true"
        
        # Set model configuration
        self._set_model_config()

        # For Bedrock provider, validate AWS credentials on startup
        if self.api_provider == "bedrock":
            if not validate_aws_credentials():
                logger.warning("AWS credentials not properly configured for Bedrock. Some features may not work.")
        
        logger.info(f"Server configured with provider: {self.api_provider}, model: {self.model}")
    
    def _set_model_config(self):
        """Set model configuration based on the selected model."""
        model_conf = (
            SONNET_3_7
            if "3-7" in self.model
            else MODEL_TO_MODEL_CONF.get(self.model, SONNET_3_5_NEW)
        )
        self.tool_version = model_conf.tool_version
        self.has_thinking = model_conf.has_thinking
        self.output_tokens = model_conf.default_output_tokens
        self.max_output_tokens = model_conf.max_output_tokens
        
        # Enable thinking by default for models that support it
        # Thinking is Claude's ability to show its reasoning process before providing a final answer
        # Only available on Claude 3.7 models
        self.thinking_budget = int(model_conf.default_output_tokens / 2) if self.has_thinking else None
        
        # Override from environment if set
        if os.getenv("TOOL_VERSION"):
            self.tool_version = os.getenv("TOOL_VERSION")
        if os.getenv("OUTPUT_TOKENS"):
            self.output_tokens = int(os.getenv("OUTPUT_TOKENS"))
        
        # Allow explicit disabling of thinking with THINKING_ENABLED=false
        # Otherwise, it's enabled by default for supported models
        if os.getenv("THINKING_ENABLED") and os.getenv("THINKING_ENABLED").lower() == "false":
            self.thinking_budget = None
        elif os.getenv("THINKING_ENABLED") and os.getenv("THINKING_ENABLED").lower() == "true":
            # If explicitly enabled but model doesn't support it, log a warning
            if not self.has_thinking:
                logger.warning(f"Thinking enabled but model {self.model} does not support thinking")
            elif os.getenv("THINKING_BUDGET"):
                # Use custom budget if specified
                self.thinking_budget = int(os.getenv("THINKING_BUDGET"))
        
        logger.info(f"Model configuration: provider={self.api_provider}, model={self.model}, thinking_enabled={self.thinking_budget is not None}")
        
    def update_from_request(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from a request."""
        provider_changed = False
        if "api_provider" in config_data:
            if self.api_provider != config_data["api_provider"]:
                provider_changed = True
            self.api_provider = config_data["api_provider"]
            # If switching providers, update model to default for that provider
            if provider_changed and "model" not in config_data:
                self.model = DEFAULT_MODEL if self.api_provider == "anthropic" else DEFAULT_BEDROCK_MODEL
                
        if "model" in config_data:
            self.model = config_data["model"]
            
        # Format model name appropriately for the provider
        self.model = format_model_name_for_provider(self.model, self.api_provider)
        
        if provider_changed or "model" in config_data:
            self._set_model_config()
            
        if "api_key" in config_data and config_data["api_key"] and self.api_provider == "anthropic":
            self.api_key = config_data["api_key"]
            save_to_storage("api_key", self.api_key)
        if "custom_system_prompt" in config_data:
            self.custom_system_prompt = config_data["custom_system_prompt"]
            save_to_storage("system_prompt", self.custom_system_prompt)
        if "only_n_most_recent_images" in config_data:
            # Cap at 90 images to stay well below the API limit of 100
            self.only_n_most_recent_images = min(int(config_data["only_n_most_recent_images"]), 90)
        if "hide_images" in config_data:
            self.hide_images = config_data["hide_images"]
        if "token_efficient_tools_beta" in config_data:
            self.token_efficient_tools_beta = config_data["token_efficient_tools_beta"]
        if "tool_version" in config_data:
            self.tool_version = config_data["tool_version"]
        if "output_tokens" in config_data:
            self.output_tokens = int(config_data["output_tokens"])
        
        # Handle thinking settings
        if "thinking_enabled" in config_data:
            if config_data["thinking_enabled"] and self.has_thinking:
                if "thinking_budget" in config_data:
                    self.thinking_budget = int(config_data["thinking_budget"])
                else:
                    # Default to half of output tokens if not specified
                    self.thinking_budget = int(self.output_tokens / 2)
            else:
                self.thinking_budget = None
        elif "thinking_budget" in config_data and self.has_thinking:
            # If only thinking_budget is provided (without thinking_enabled),
            # assume thinking is enabled
            self.thinking_budget = int(config_data["thinking_budget"])
            
        logger.info(f"Configuration updated: provider={self.api_provider}, model={self.model}, thinking_enabled={self.thinking_budget is not None}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "api_provider": self.api_provider,
            "model": self.model,
            "custom_system_prompt": self.custom_system_prompt,
            "only_n_most_recent_images": self.only_n_most_recent_images,
            "hide_images": self.hide_images,
            "token_efficient_tools_beta": self.token_efficient_tools_beta,
            "tool_version": self.tool_version,
            "output_tokens": self.output_tokens,
            "has_thinking": self.has_thinking,
            "thinking_enabled": self.thinking_budget is not None,
            "thinking_budget": self.thinking_budget,
            "available_tool_versions": list(get_args(ToolVersion)),
            "available_models": list(MODEL_TO_MODEL_CONF.keys()) + ["claude-3-5-sonnet-20240620"],
            "available_providers": ["anthropic", "bedrock"]
        }
        
        # Add provider-specific information
        if self.api_provider == "bedrock":
            # Add Bedrock-specific information
            config_dict["bedrock_models"] = [
                "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            ]
            config_dict["aws_region"] = os.environ.get("AWS_REGION", "Not set")
            
            # Check AWS credentials status
            has_credentials = validate_aws_credentials()
            config_dict["aws_credentials_valid"] = has_credentials
        elif self.api_provider == "anthropic":
            # Add Anthropic-specific information
            config_dict["has_api_key"] = self.api_key is not None
        
        return config_dict

class ChatState:
    def __init__(self):
        self.messages = []
        self.tools = {}
        self.responses = {}
        self.request_id = datetime.now().isoformat()
        self.config = ServerConfig()
        self.in_sampling_loop = False
        
        # Validate credentials based on provider
        if self.config.api_provider == "bedrock":
            if not validate_aws_credentials():
                logger.warning("AWS credentials not properly configured")
                # Don't raise an exception here, just log a warning
        elif self.config.api_provider == "anthropic" and not self.config.api_key:
            logger.warning("Anthropic API key not set")
        
        logger.info(f"Created new ChatState with request_id: {self.request_id}, provider: {self.config.api_provider}")

    def cleanup_history(self, max_images=90):
        """
        Ensure that the total number of images in the message history doesn't exceed max_images.
        This helps prevent "Too much media" errors from the Anthropic API.
        """
        if not self.messages:
            return
            
        # Count total images in the conversation history
        total_images = 0
        image_blocks = []
        
        for message in self.messages:
            if not isinstance(message.get("content"), list):
                continue
                
            for block in message["content"]:
                if isinstance(block, dict):
                    # Count images in tool results
                    if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                        for content in block["content"]:
                            if isinstance(content, dict) and content.get("type") == "image":
                                total_images += 1
                                image_blocks.append((message, block, content))
                    # Count direct images
                    elif block.get("type") == "image":
                        total_images += 1
                        image_blocks.append((message, None, block))
        
        # If we're over the limit, remove oldest images first
        images_to_remove = max(0, total_images - max_images)
        logger.info(f"Chat history has {total_images} images, removing {images_to_remove}")
        
        for i in range(images_to_remove):
            if i >= len(image_blocks):
                break
                
            message, tool_result, image = image_blocks[i]
            
            if tool_result:
                # Image is inside a tool result
                tool_result["content"].remove(image)
                # If tool result is now empty, remove it
                if not tool_result["content"]:
                    message["content"].remove(tool_result)
            else:
                # Image is directly in message content
                message["content"].remove(image)
            
        # Clean up empty messages
        self.messages = [msg for msg in self.messages if msg.get("content")]
        
        logger.info(f"Cleaned up history, now has {total_images - images_to_remove} images")

    def maybe_add_interruption_blocks(self):
        """Add interruption blocks to the conversation if needed."""
        if not self.in_sampling_loop:
            return []
        
        # If this function is called while we're in the sampling loop, we can assume that the previous sampling loop was interrupted
        # and we should annotate the conversation with additional context for the model and heal any incomplete tool use calls
        result = []
        try:
            last_message = self.messages[-1]
            if last_message["role"] == "assistant" and isinstance(last_message["content"], list):
                previous_tool_use_ids = [
                    block["id"] for block in last_message["content"] 
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ]
                for tool_use_id in previous_tool_use_ids:
                    self.tools[tool_use_id] = ToolResult(error=INTERRUPT_TOOL_ERROR)
                    result.append({
                        "tool_use_id": tool_use_id,
                        "type": "tool_result",
                        "content": INTERRUPT_TOOL_ERROR,
                        "is_error": True
                    })
        except (IndexError, KeyError) as e:
            logger.error(f"Error adding interruption blocks: {e}")
        
        result.append({"type": "text", "text": INTERRUPT_TEXT})
        return result

    @contextmanager
    def track_sampling_loop(self):
        """Track when we're in a sampling loop."""
        self.in_sampling_loop = True
        try:
            yield
        finally:
            self.in_sampling_loop = False

def write_stream_chunked(handler, data: Dict[str, Any], event_type: str = None):
    """Write an event to the stream, chunking large base64 data if needed."""
    try:
        # Handle large base64 data more efficiently
        if isinstance(data, dict) and "content" in data:
            content = data["content"]
            if isinstance(content, dict) and content.get("type") == "image":
                # Split large base64 data into chunks
                source = content.get("source", {})
                if source.get("type") == "base64" and len(source.get("data", "")) > 1024 * 1024:
                    chunks = [source["data"][i:i + 1024*1024] for i in range(0, len(source["data"]), 1024*1024)]
                    for i, chunk in enumerate(chunks):
                        chunk_data = data.copy()
                        chunk_data["content"] = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": source.get("media_type", "image/png"),
                                "data": chunk
                            },
                            "chunk": i + 1,
                            "total_chunks": len(chunks)
                        }
                        message = f"data: {json.dumps(chunk_data)}\n"
                        if event_type:
                            message = f"event: {event_type}\n{message}"
                        message += "\n"
                        handler.wfile.write(message.encode('utf-8'))
                        handler.wfile.flush()
                    return

        message = f"data: {json.dumps(data)}\n"
        if event_type:
            message = f"event: {event_type}\n{message}"
        message += "\n"
        handler.wfile.write(message.encode('utf-8'))
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        # Client disconnected
        return False
    except Exception as e:
        logger.error(f"Error in write_stream: {str(e)}")
        # Don't re-raise, just log the error
        return False
    return True

class APIHandler(BaseHTTPRequestHandler):
    # Class-level chat state to persist across requests
    _chat_state = ChatState()
    
    def __init__(self, *args, **kwargs):
        self.client_disconnected = False
        logger.info(f"Using existing ChatState with request_id: {self._chat_state.request_id}")
        super().__init__(*args, **kwargs)
    
    def handle_one_request(self):
        try:
            logger.debug(f"[{self._chat_state.request_id}] Handling new request")
            return super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError) as e:
            self.client_disconnected = True
            logger.error(f"[{self._chat_state.request_id}] Connection error in handle_one_request: {str(e)}")
            return
        except Exception as e:
            logger.error(f"[{self._chat_state.request_id}] Error in handle_one_request: {str(e)}")
            return

    def write_stream(self, data: Dict[str, Any], event_type: str = None):
        if self.client_disconnected:
            logger.warning(f"[{self._chat_state.request_id}] Attempted to write to disconnected client")
            return False
        return write_stream_chunked(self, data, event_type)

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Max-Age', '3600')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        
    def end_headers(self):
        self.send_cors_headers()
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def output_callback(self, content: str | Dict[str, Any]) -> None:
        """Handle assistant output by streaming it to client."""
        if self.client_disconnected:
            return
            
        if isinstance(content, str):
            self.write_stream({"type": "assistant_message", "content": content}, "message")
        else:
            # Handle thinking content
            if isinstance(content, dict) and content.get("type") == "thinking":
                thinking_content = content.get("thinking", "")
                self.write_stream({
                    "type": "thinking",
                    "content": thinking_content
                }, "thinking")
                return
                
            # Check if this is a tool result with an image
            if isinstance(content, dict) and content.get("type") == "tool_result":
                tool_content = content.get("content", [])
                if isinstance(tool_content, list):
                    for item in tool_content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            if not self.write_stream({
                                "type": "assistant_content",
                                "content": {
                                    "type": "image",
                                    "source": item.get("source")
                                }
                            }, "message"):
                                return
                            continue
            self.write_stream({"type": "assistant_content", "content": content}, "message")

    def tool_output_callback(self, tool_output: ToolResult, tool_id: str) -> None:
        """Handle a tool output by streaming it to client."""
        if self.client_disconnected:
            return
            
        self._chat_state.tools[tool_id] = tool_output
        output_data = {
            "type": "tool_output",
            "tool_id": tool_id,
            "output": {
                "text": tool_output.output,
                "error": tool_output.error,
                "base64_image": tool_output.base64_image if not self._chat_state.config.hide_images else None,
                "system": tool_output.system
            }
        }
        if not self.write_stream(output_data, "tool"):
            return

        # If there's a screenshot and images aren't hidden, send it as a separate message
        if tool_output.base64_image and not self._chat_state.config.hide_images:
            self.write_stream({
                "type": "assistant_content",
                "content": {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": tool_output.base64_image
                    }
                }
            }, "message")

    def api_response_callback(self, request: httpx.Request, response: Optional[httpx.Response | object] = None, error: Optional[Exception] = None) -> None:
        """Handle API response by streaming it to client."""
        if request and hasattr(request, 'url'):
            response_id = datetime.now().isoformat()
            self._chat_state.responses[str(request.url)] = (request, response)
            self.write_stream({
                "type": "api_response",
                "response_id": response_id,
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "body": request.read().decode() if hasattr(request, 'read') else None
                },
                "response": response.text if isinstance(response, httpx.Response) else str(response) if response else None,
                "error": str(error) if error else None
            }, "api")
            
            # Handle errors, especially rate limits
            if error:
                self.render_error(error)
    
    def render_error(self, error: Exception):
        """Render an error to the client."""
        if isinstance(error, RateLimitError):
            body = "You have been rate limited."
            if hasattr(error, 'response') and hasattr(error.response, 'headers'):
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    body += f" Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS). See API documentation for more details."
            if hasattr(error, 'message'):
                body += f"\n\n{error.message}"
        else:
            body = str(error)
            body += "\n\n**Traceback:**"
            lines = "\n".join(traceback.format_exception(error))
            body += f"\n\n```{lines}```"
        
        save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
        
        self.write_stream({
            "type": "error",
            "error": body,
            "error_class": error.__class__.__name__,
            "status": "error"
        }, "error")
        
    async def handle_chat(self, message):
        try:
            if self.client_disconnected:
                return
            
            # Check if we need to add interruption blocks
            interruption_blocks = self._chat_state.maybe_add_interruption_blocks()
                
            # Add user message to history
            self._chat_state.messages.append({
                "role": "user",
                "content": [
                    *interruption_blocks,
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            })
            
            if not self.write_stream({
                "type": "user_message",
                "content": message
            }, "message"):
                return
            
            config = self._chat_state.config
            
            # Clean up history to ensure we don't exceed image limits
            max_images = min(90, config.only_n_most_recent_images or 90)
            self._chat_state.cleanup_history(max_images=max_images)
            
            # Check credentials based on provider
            if config.api_provider == "bedrock":
                if not validate_aws_credentials():
                    raise Exception("AWS credentials not properly configured for Bedrock access")
            elif config.api_provider == "anthropic" and not config.api_key:
                raise Exception("Anthropic API key not provided")
            
            # Run the agent sampling loop with tracking
            with self._chat_state.track_sampling_loop():
                self._chat_state.messages = await sampling_loop(
                    model=config.model,
                    provider=APIProvider(config.api_provider),
                    messages=self._chat_state.messages,
                    api_key=config.api_key,
                    system_prompt_suffix=config.custom_system_prompt,
                    output_callback=self.output_callback,
                    tool_output_callback=self.tool_output_callback,
                    api_response_callback=self.api_response_callback,
                    only_n_most_recent_images=config.only_n_most_recent_images,
                    tool_version=config.tool_version,
                    max_tokens=config.output_tokens,
                    thinking_budget=config.thinking_budget,
                    token_efficient_tools_beta=config.token_efficient_tools_beta
                )
            
            # Get the last assistant message
            for msg in reversed(self._chat_state.messages):
                if msg["role"] == "assistant":
                    # Format response
                    if isinstance(msg["content"], str):
                        response_text = msg["content"]
                    else:
                        response_text = "\n".join(
                            block["text"] for block in msg["content"] 
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    
                    self.write_stream({
                        "type": "final_response",
                        "role": "assistant",
                        "content": response_text,
                        "status": "success"
                    }, "complete")
                    return
            
            raise Exception("No assistant response found")
            
        except Exception as e:
            logger.error(f"Error in handle_chat: {str(e)}")
            self.render_error(e)
    
    def handle_config_update(self, config_data):
        """Handle configuration update request."""
        try:
            # Update configuration
            self._chat_state.config.update_from_request(config_data)
            
            # Return updated configuration
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "success",
                "config": self._chat_state.config.to_dict()
            }).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"[{self._chat_state.request_id}] Error updating configuration: {str(e)}")
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "error",
                "error": str(e)
            }).encode('utf-8'))
    
    def handle_get_config(self):
        """Handle get configuration request."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "success",
            "config": self._chat_state.config.to_dict()
        }).encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/api/config':
            self.handle_get_config()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/chat':
            logger.info(f"[{self._chat_state.request_id}] Received POST request to /api/chat")
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
                message = request_data.get('message', '')
                
                logger.debug(f"[{self._chat_state.request_id}] Received message: {message[:100]}...")
                
                # Send headers for event stream
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')  # Changed to 'close' instead of 'keep-alive'
                self.end_headers()

                # Use the existing event loop if available, otherwise create a new one
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    logger.warning(f"[{self._chat_state.request_id}] Event loop was closed, creating new one")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                try:
                    logger.debug(f"[{self._chat_state.request_id}] Starting chat handler")
                    loop.run_until_complete(self.handle_chat(message))
                    logger.debug(f"[{self._chat_state.request_id}] Chat handler completed")
                except Exception as e:
                    logger.error(f"[{self._chat_state.request_id}] Error in chat handler: {str(e)}")
                    self.render_error(e)
                finally:
                    logger.info(f"[{self._chat_state.request_id}] Closing response")
                    self.wfile.flush()
                
            except Exception as e:
                logger.error(f"[{self._chat_state.request_id}] Error in do_POST: {str(e)}")
                self.send_response(500)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.write_stream({
                    "type": "error",
                    "error": str(e),
                    "status": "error"
                }, "error")
        elif self.path == '/api/config':
            # Handle configuration update
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                config_data = json.loads(post_data.decode('utf-8'))
                self.handle_config_update(config_data)
            except Exception as e:
                logger.error(f"[{self._chat_state.request_id}] Error handling config update: {str(e)}")
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error": str(e)
                }).encode('utf-8'))
        elif self.path == '/api/reset':
            # Handle reset request
            try:
                # Create a new chat state
                self.__class__._chat_state = ChatState()
                
                # Return success
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "message": "Chat state reset successfully"
                }).encode('utf-8'))
            except Exception as e:
                logger.error(f"Error resetting chat state: {str(e)}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error": str(e)
                }).encode('utf-8'))
        else:
            logger.warning(f"[{self._chat_state.request_id}] Invalid path requested: {self.path}")
            self.send_error(404, "Not Found")

    def finish(self):
        """Clean up any resources when the request is done"""
        logger.info(f"[{self._chat_state.request_id}] Finishing request and cleaning up resources")
        self.client_disconnected = True
        # Remove the chat state cleanup since we want to persist it
        try:
            super().finish()
            logger.info(f"[{self._chat_state.request_id}] Request finished successfully")
        except Exception as e:
            logger.error(f"[{self._chat_state.request_id}] Error in finish: {str(e)}")

class ThreadedHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.daemon_threads = True

def run_server(port=8083):
    try:
        print(f"Starting API server initialization on port {port}...")
        
        # Check credentials based on default provider
        default_provider = os.getenv("API_PROVIDER", "bedrock")
        if default_provider == "bedrock":
            if not validate_aws_credentials():
                print("WARNING: AWS credentials not properly configured for Bedrock")
                print("Make sure AWS_REGION, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY are set")
                print("Or ensure AWS credentials file is properly configured")
        elif default_provider == "anthropic":
            try:
                get_api_key()
            except ValueError:
                print("WARNING: Anthropic API key not set")
                print("Set ANTHROPIC_API_KEY environment variable or create ~/.anthropic/api_key file")
        
        server_address = ("0.0.0.0", port)  # Explicitly bind to all interfaces
        httpd = ThreadedHTTPServer(server_address, APIHandler)
        print(f"API Server is running at http://0.0.0.0:{port}")
        print(f"Using {default_provider.upper()} as the default provider")
        print("Server is ready to handle requests")
        httpd.serve_forever()
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Initializing HTTP server...")
    run_server()
