# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Pydantic models for Anthropic Messages API.

Defines data schemas for requests and responses compatible with
Anthropic's Messages API specification.

Reference: https://docs.anthropic.com/en/api/messages
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, model_validator


# ==================================================================================================
# Content Block Models
# ==================================================================================================


class TextContentBlock(BaseModel):
    """
    Text content block in Anthropic format.

    Used in both requests and responses for text content.
    """

    type: Literal["text"] = "text"
    text: str


class ThinkingContentBlock(BaseModel):
    """
    Thinking content block in Anthropic format.

    Represents the model's reasoning/thinking process.
    Used when extended thinking is enabled.

    Attributes:
        type: Always "thinking"
        thinking: The thinking/reasoning content
        signature: Cryptographic signature for verification (placeholder in our case)
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class ToolUseContentBlock(BaseModel):
    """
    Tool use content block in Anthropic format.

    Represents a tool call made by the assistant.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolReferenceContentBlock(BaseModel):
    """
    Tool reference content block (Claude Code deferred tools).

    Sent by Claude Code v2.1.69+ inside tool_result blocks to indicate
    which tools were loaded via the ToolSearch deferred tool mechanism.
    """

    type: Literal["tool_reference"] = "tool_reference"
    tool_name: str

    model_config = {"extra": "allow"}


class ServerToolUseContentBlock(BaseModel):
    """
    Anthropic server-side tool call (web_search, web_fetch, code_execution).

    These blocks live on the assistant message. They are not client tool_use
    and must not be forwarded to Kiro as toolUses.
    """

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class WebSearchToolResultContentBlock(BaseModel):
    """
    Result of an Anthropic web_search server tool.

    Appears on the same assistant message as the matching server_tool_use.
    """

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str
    content: Any = None

    model_config = {"extra": "allow"}


class UnknownContentBlock(BaseModel):
    """
    Catch-all for Anthropic content types the gateway does not model yet.

    Kept last in ContentBlock so known Literal types still win.
    """

    type: str

    model_config = {"extra": "allow"}


class ToolResultContentBlock(BaseModel):
    """
    Tool result content block in Anthropic format.

    Represents the result of a tool call, sent by the user.
    Tool results can contain text, images, tool references, or a mix.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[
        Union[str, List[Union["TextContentBlock", "ImageContentBlock", "ToolReferenceContentBlock"]]]
    ] = None
    is_error: Optional[bool] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Image Content Block Models
# ==================================================================================================


class Base64ImageSource(BaseModel):
    """
    Base64-encoded image source in Anthropic format.

    Attributes:
        type: Always "base64"
        media_type: MIME type (e.g., "image/jpeg", "image/png", "image/gif", "image/webp")
        data: Base64-encoded image data
    """

    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class URLImageSource(BaseModel):
    """
    URL-based image source in Anthropic format.

    Note: URL images require fetching and converting to base64 for Kiro API.
    Currently logged as warning and skipped.

    Attributes:
        type: Always "url"
        url: HTTP(S) URL to the image
    """

    type: Literal["url"] = "url"
    url: str


class ImageContentBlock(BaseModel):
    """
    Image content block in Anthropic format.

    Represents an image in a message. Supports both base64-encoded
    images and URL references.

    Attributes:
        type: Always "image"
        source: Image source (base64 or URL)
    """

    type: Literal["image"] = "image"
    source: Union[Base64ImageSource, URLImageSource]


# Union type for all content blocks (including images and thinking)
ContentBlock = Union[
    TextContentBlock,
    ThinkingContentBlock,
    ImageContentBlock,
    ToolUseContentBlock,
    ToolResultContentBlock,
    ToolReferenceContentBlock,
    ServerToolUseContentBlock,
    WebSearchToolResultContentBlock,
    UnknownContentBlock,
]


# ==================================================================================================
# Message Models
# ==================================================================================================


class AnthropicMessage(BaseModel):
    """
    Message in Anthropic format.

    Attributes:
        role: Message role (user or assistant)
        content: Message content (string or list of content blocks)
    """

    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]

    model_config = {"extra": "allow"}


# ==================================================================================================
# Tool Models
# ==================================================================================================


class AnthropicTool(BaseModel):
    """
    Tool definition in Anthropic format.
    
    Supports both user-defined tools and server-side tools (Anthropic):
    - User-defined tools: require input_schema
    - Server-side tools: use type field (e.g., "web_search_20250305")
    
    Attributes:
        type: Tool type for server-side tools (e.g., "web_search_20250305")
        name: Tool name (must match pattern ^[a-zA-Z0-9_-]{1,64}$)
        description: Tool description (optional but recommended)
        input_schema: JSON Schema for tool parameters (required for user-defined tools)
        max_uses: Maximum uses per conversation (server-side tools, optional)
        allowed_domains: Allowed domains for web_search (optional)
        blocked_domains: Blocked domains for web_search (optional)
        user_location: User location for web_search (optional)
    """
    
    # Server-side tool fields (Anthropic spec)
    type: Optional[str] = None
    
    # Common fields
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None  # Now optional for server-side tools
    
    # Server-side tool parameters (Anthropic spec - accepted but not enforced)
    max_uses: Optional[int] = None
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    user_location: Optional[Dict[str, Any]] = None
    
    model_config = {"extra": "allow"}  # Forward compatibility
    
    @model_validator(mode="after")
    def validate_tool_consistency(self):
        """Validate that user-defined tools have input_schema."""
        is_server_side = self.type is not None
        
        if not is_server_side:
            # User-defined tool: input_schema is required
            if self.input_schema is None:
                raise ValueError(
                    "input_schema is required for user-defined tools "
                    "(those without a 'type' field)"
                )
        return self


class ToolChoiceAuto(BaseModel):
    """Auto tool choice - model decides whether to use tools."""

    type: Literal["auto"] = "auto"


class ToolChoiceAny(BaseModel):
    """Any tool choice - model must use at least one tool."""

    type: Literal["any"] = "any"


class ToolChoiceTool(BaseModel):
    """Specific tool choice - model must use the specified tool."""

    type: Literal["tool"] = "tool"
    name: str


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool]


# ==================================================================================================
# Request Models
# ==================================================================================================


class SystemContentBlock(BaseModel):
    """
    System content block for prompt caching.

    Anthropic API supports system as a list of content blocks
    with optional cache_control for prompt caching.
    """

    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# System can be a string or list of content blocks (for prompt caching)
SystemPrompt = Union[str, List[SystemContentBlock], List[Dict[str, Any]]]


def _fold_system_role_messages(data: Any) -> Any:
    """
    Fold ``role: "system"`` entries from ``messages`` into the top-level ``system`` field.

    Claude Code 2.1.156+ injects session/hook context (``<system-reminder>`` blocks,
    current date, etc.) as ``role: "system"`` entries inside the ``messages`` array
    instead of the top-level ``system`` field. The Anthropic Messages API rejects
    those entries with a 422 because ``messages[].role`` must be ``"user"`` or
    ``"assistant"``. This helper salvages the request by lifting their content into
    ``system`` so the proxy still produces a spec-compliant payload.

    The helper preserves cache markers: each folded message becomes a
    ``SystemContentBlock``-shaped dict, and existing ``system`` content blocks are
    kept intact. A pure-string ``system`` is upgraded to a list only when at least
    one folded message contributes a block (so the simple "no folding needed" path
    leaves the request untouched).

    Args:
        data: Raw request payload as a dict (running in pydantic ``mode="before"``).

    Returns:
        The same dict, mutated in place when folding happened. Non-dict inputs are
        returned untouched so pydantic can produce its own validation error.
    """
    if not isinstance(data, dict):
        return data

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return data

    kept_messages: List[Any] = []
    folded_blocks: List[Dict[str, Any]] = []
    folded_count = 0

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

        if role != "system":
            kept_messages.append(msg)
            continue

        folded_count += 1
        for block in _iter_system_blocks_from_content(content):
            folded_blocks.append(block)

    if folded_count == 0:
        return data

    if folded_blocks:
        existing_blocks = _normalize_existing_system(data.get("system"))
        data["system"] = existing_blocks + folded_blocks
    data["messages"] = kept_messages

    logger.debug(
        f"Folded {folded_count} role='system' message(s) into top-level system field "
        f"({len(folded_blocks)} block(s) added; system field "
        f"{'updated' if folded_blocks else 'left unchanged because all blocks were empty'})"
    )
    return data


def _iter_system_blocks_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Convert a folded message's ``content`` to a list of ``SystemContentBlock``-shaped dicts.

    Accepts the two Anthropic content shapes (``str`` or ``list`` of blocks) plus
    pydantic content-block instances. Image / tool_use / tool_result blocks inside
    a ``role: "system"`` message are silently dropped because they have no meaning
    as system context.

    Args:
        content: ``content`` value from a ``role: "system"`` message.

    Returns:
        Zero or more dicts of the form ``{"type": "text", "text": "..."}``, with
        ``cache_control`` preserved when present on the source block.
    """
    if content is None:
        return []

    if isinstance(content, str):
        if not content:
            return []
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return []

    blocks: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") != "text":
                continue
            text = item.get("text", "")
            if not isinstance(text, str) or not text:
                continue
            block: Dict[str, Any] = {"type": "text", "text": text}
            cache_control = item.get("cache_control")
            if cache_control is not None:
                block["cache_control"] = cache_control
            blocks.append(block)
        else:
            block_type = getattr(item, "type", None)
            if block_type != "text":
                continue
            text = getattr(item, "text", "")
            if not isinstance(text, str) or not text:
                continue
            block = {"type": "text", "text": text}
            cache_control = getattr(item, "cache_control", None)
            if cache_control is not None:
                block["cache_control"] = cache_control
            blocks.append(block)
    return blocks


def _normalize_existing_system(system: Any) -> List[Dict[str, Any]]:
    """
    Normalize the existing top-level ``system`` value to a list of block-shaped dicts.

    Args:
        system: Current value of ``system`` from the raw request payload.

    Returns:
        A list of ``{"type": "text", ...}`` dicts. Empty when ``system`` is missing
        or empty. Non-text blocks (rare; not part of the Anthropic spec) are
        passed through unchanged so we never drop user content silently.
    """
    if system is None:
        return []
    if isinstance(system, str):
        if not system:
            return []
        return [{"type": "text", "text": system}]
    if isinstance(system, list):
        normalized: List[Dict[str, Any]] = []
        for item in system:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "model_dump"):
                normalized.append(item.model_dump(exclude_none=True))
            else:
                text = getattr(item, "text", None)
                if text is None:
                    continue
                block: Dict[str, Any] = {"type": "text", "text": text}
                cache_control = getattr(item, "cache_control", None)
                if cache_control is not None:
                    block["cache_control"] = cache_control
                normalized.append(block)
        return normalized
    return []


class AnthropicMessagesRequest(BaseModel):
    """
    Request to Anthropic Messages API (/v1/messages).

    Attributes:
        model: Model ID (e.g., "claude-sonnet-4-5")
        messages: List of conversation messages
        max_tokens: Maximum tokens in response (required)
        system: System prompt (optional, string or list of content blocks for caching)
        stream: Whether to stream the response
        tools: List of available tools
        tool_choice: Tool selection strategy
        temperature: Sampling temperature (0-1)
        top_p: Top-p sampling
        top_k: Top-k sampling
        stop_sequences: Custom stop sequences
        metadata: Request metadata
    """

    model: str
    messages: List[AnthropicMessage] = Field(min_length=1)
    max_tokens: int

    # Optional parameters - system can be string or list of content blocks
    system: Optional[SystemPrompt] = None
    stream: bool = False

    # Extended thinking (official Anthropic parameter)
    thinking: Optional[Dict[str, Any]] = None

    # Tools
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[ToolChoice, Dict[str, Any]]] = None

    # Sampling parameters
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)

    # Other parameters
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _fold_system_role_messages(cls, data: Any) -> Any:
        """Fold ``role: "system"`` messages into the top-level ``system`` field.

        See :func:`_fold_system_role_messages` for the full rationale (Claude
        Code 2.1.156+ injects session context as ``role: "system"`` entries
        that the Anthropic spec rejects).
        """
        return _fold_system_role_messages(data)


class AnthropicCountTokensRequest(BaseModel):
    """
    Request to Anthropic Count Tokens API (/v1/messages/count_tokens).
    
    Similar to AnthropicMessagesRequest but without generation parameters.
    Used to estimate token count before making actual request.
    
    Attributes:
        model: Model ID (e.g., "claude-sonnet-4-5")
        messages: List of conversation messages
        system: System prompt (optional, string or list of content blocks)
        tools: List of available tools
    """
    
    model: str
    messages: List[AnthropicMessage] = Field(min_length=1)

    # Optional parameters - only those that affect token count
    system: Optional[SystemPrompt] = None
    tools: Optional[List[AnthropicTool]] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _fold_system_role_messages(cls, data: Any) -> Any:
        """Fold ``role: "system"`` messages into the top-level ``system`` field.

        Same rationale as :class:`AnthropicMessagesRequest`. The
        ``/v1/messages/count_tokens`` endpoint accepts the same payload shape
        and would otherwise 422 on Claude Code 2.1.156+ requests.
        """
        return _fold_system_role_messages(data)


# ==================================================================================================
# Response Models
# ==================================================================================================


class AnthropicUsage(BaseModel):
    """
    Token usage information in Anthropic format.

    Attributes:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_read_input_tokens: Tokens read from prompt cache (only forwarded when explicitly returned by upstream Kiro API)
        cache_creation_input_tokens: Tokens used to create prompt cache (only forwarded when explicitly returned by upstream Kiro API)
    """

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None

    model_config = {"extra": "allow"}


class AnthropicMessagesResponse(BaseModel):
    """
    Response from Anthropic Messages API (non-streaming).

    Attributes:
        id: Unique message ID
        type: Always "message"
        role: Always "assistant"
        content: List of content blocks (may include thinking, text, tool_use)
        model: Model used
        stop_reason: Why generation stopped
        stop_sequence: Stop sequence that triggered stop (if any)
        usage: Token usage information
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[ThinkingContentBlock, TextContentBlock, ToolUseContentBlock]]
    model: str
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# ==================================================================================================
# Streaming Event Models
# ==================================================================================================


class MessageStartEvent(BaseModel):
    """
    Event sent at the start of a message stream.

    Contains the initial message object with empty content.
    """

    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any]


class ContentBlockStartEvent(BaseModel):
    """
    Event sent at the start of a content block.

    Attributes:
        index: Index of the content block
        content_block: Initial content block (with empty text for text blocks)
    """

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Dict[str, Any]


class TextDelta(BaseModel):
    """Delta for text content."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseModel):
    """Delta for thinking content."""

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class InputJsonDelta(BaseModel):
    """Delta for tool input JSON."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    """
    Event sent when content block is updated.

    Attributes:
        index: Index of the content block being updated
        delta: The delta update (text_delta, thinking_delta, or input_json_delta)
    """

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[TextDelta, ThinkingDelta, InputJsonDelta, Dict[str, Any]]


class ContentBlockStopEvent(BaseModel):
    """
    Event sent when a content block is complete.
    """

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaUsage(BaseModel):
    """Usage information in message_delta event."""

    output_tokens: int


class MessageDeltaEvent(BaseModel):
    """
    Event sent near the end of the stream with final message data.

    Attributes:
        delta: Contains stop_reason and stop_sequence
        usage: Output token count
    """

    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: MessageDeltaUsage


class MessageStopEvent(BaseModel):
    """
    Event sent at the end of the message stream.
    """

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """
    Ping event sent periodically to keep connection alive.
    """

    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """
    Error event sent when an error occurs during streaming.
    """

    type: Literal["error"] = "error"
    error: Dict[str, Any]


# Union of all streaming events
StreamingEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent,
]


# ==================================================================================================
# Error Models
# ==================================================================================================


class AnthropicErrorDetail(BaseModel):
    """
    Error detail in Anthropic format.
    """

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """
    Error response in Anthropic format.
    """

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
