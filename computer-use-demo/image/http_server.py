import os
import socket
import json
import asyncio
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial
import anthropic
from computer_use_demo.loop import sampling_loop
from computer_use_demo.tools import ToolResult
from typing import Any, Dict, Optional
import httpx
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API key from environment variable
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

class ChatState:
    def __init__(self):
        self.messages = []
        self.tools = {}
        self.responses = {}
        self.request_id = datetime.now().isoformat()
        logger.info(f"Created new ChatState with request_id: {self.request_id}")

def write_stream(handler, data: Dict[str, Any], event_type: str = None):
    """Write an event to the stream"""
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
        return
    except Exception as e:
        print(f"Error in write_stream: {str(e)}")
        # Don't re-raise, just log the error
        return

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
        try:
            message = f"data: {json.dumps(data)}\n"
            if event_type:
                message = f"event: {event_type}\n{message}"
            message += "\n"
            logger.debug(f"[{self._chat_state.request_id}] Writing stream data type: {event_type}")
            self.wfile.write(message.encode('utf-8'))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError) as e:
            self.client_disconnected = True
            logger.error(f"[{self._chat_state.request_id}] Connection error in write_stream: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"[{self._chat_state.request_id}] Error in write_stream: {str(e)}")
            return False

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
                "base64_image": tool_output.base64_image,
                "system": tool_output.system
            }
        }
        if not self.write_stream(output_data, "tool"):
            return

        # If there's a screenshot, send it as a separate message
        if tool_output.base64_image:
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
        
    async def handle_chat(self, message):
        try:
            if self.client_disconnected:
                return
                
            # Add user message to history
            self._chat_state.messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": message
                }]
            })
            
            if not self.write_stream({
                "type": "user_message",
                "content": message
            }, "message"):
                return
            
            # Run the agent sampling loop
            self._chat_state.messages = await sampling_loop(
                model="claude-3-5-sonnet-20241022",
                provider="anthropic",
                messages=self._chat_state.messages,
                api_key=ANTHROPIC_API_KEY,
                system_prompt_suffix="",
                output_callback=self.output_callback,
                tool_output_callback=self.tool_output_callback,
                api_response_callback=self.api_response_callback,
                only_n_most_recent_images=3
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
            print(f"Error in handle_chat: {str(e)}")
            error_response = {
                "type": "error",
                "content": f"Error: {str(e)}",
                "status": "error"
            }
            self.write_stream(error_response, "error")
    
    def do_POST(self):
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
                    self.write_stream({
                        "type": "error",
                        "error": str(e),
                        "status": "error"
                    }, "error")
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
        server_address = ("0.0.0.0", port)  # Explicitly bind to all interfaces
        httpd = ThreadedHTTPServer(server_address, APIHandler)
        print(f"API Server is running at http://0.0.0.0:{port}")
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
