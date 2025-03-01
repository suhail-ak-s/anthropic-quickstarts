# HTTP Server API Documentation

The Computer Use Demo includes an HTTP server implementation that provides a REST API for interacting with Claude. This document describes the available endpoints and their parameters.

## Overview

The HTTP server provides the following endpoints:

- `/api/chat` - Send messages to Claude and receive streaming responses
- `/api/config` - Get or update the server configuration
- `/api/reset` - Reset the chat state

## Chat API

### POST `/api/chat`

Send a message to Claude and receive a streaming response using Server-Sent Events (SSE).

#### Request

```json
{
  "message": "Your message to Claude"
}
```

#### Response

The response is a stream of Server-Sent Events with the following event types:

- `message` - Regular text messages from Claude
- `thinking` - Claude's thinking process (enabled by default for Claude 3.7 models)
- `tool` - Tool outputs (e.g., screenshots, command results)
- `error` - Error messages
- `complete` - Final response

Example event:

```
event: message
data: {"type": "assistant_message", "content": "I'll help you with that."}
```

## Configuration API

### GET `/api/config`

Get the current server configuration.

#### Response

```json
{
  "status": "success",
  "config": {
    "model": "claude-3-7-sonnet-20250219",
    "custom_system_prompt": "",
    "only_n_most_recent_images": 3,
    "hide_images": false,
    "token_efficient_tools_beta": false,
    "tool_version": "computer_use_20250124",
    "output_tokens": 16384,
    "has_thinking": true,
    "thinking_enabled": true,
    "thinking_budget": 8192,
    "available_tool_versions": ["computer_use_20241022", "computer_use_20250124"],
    "available_models": ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20240620"]
  }
}
```

### POST `/api/config`

Update the server configuration.

#### Request

```json
{
  "model": "claude-3-7-sonnet-20250219",
  "custom_system_prompt": "You are a helpful assistant...",
  "only_n_most_recent_images": 3,
  "hide_images": false,
  "token_efficient_tools_beta": false,
  "tool_version": "computer_use_20250124",
  "output_tokens": 16384,
  "thinking_enabled": true,
  "thinking_budget": 8192
}
```

All fields are optional. Only the fields you include will be updated.

#### Response

```json
{
  "status": "success",
  "config": {
    // Updated configuration (same format as GET /api/config)
  }
}
```

## Reset API

### POST `/api/reset`

Reset the chat state, clearing all messages and starting a new conversation.

#### Response

```json
{
  "status": "success",
  "message": "Chat state reset successfully"
}
```

## Thinking Capability

Claude 3.7 models support a "thinking" capability that allows Claude to show its reasoning process before providing a final answer. This feature is **enabled by default** for Claude 3.7 models in the HTTP server implementation.

### Controlling Thinking

You can control the thinking capability through:

1. **Environment Variables**:
   - `THINKING_ENABLED=false` - Disable thinking (it's enabled by default)
   - `THINKING_BUDGET=8192` - Set a custom token budget for thinking

2. **Configuration API**:
   ```json
   {
     "thinking_enabled": true,
     "thinking_budget": 8192
   }
   ```

3. **Client-Side Handling**:
   The client will receive thinking content as Server-Sent Events with the event type "thinking":
   ```
   event: thinking
   data: {"type": "thinking", "content": "I need to analyze this problem step by step..."}
   ```

### Benefits of Thinking

1. **Transparency**: Users can see Claude's reasoning process
2. **Debugging**: Helps identify where Claude might be making mistakes
3. **Education**: Shows how Claude approaches complex problems
4. **Trust**: Builds user confidence by showing the work

The thinking capability is particularly useful for complex computer use tasks where Claude needs to reason through multiple steps or consider different approaches. 