# Complete Code Walk: `openai_parse_chat_completion_request(...)` - Messages

Source:
[`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L1719)

<strong><em><a href="0003_A_2_openai_parse_chat_completion_request_media.md"><span style="color:#0b63ce">Continue to media admission: 0003_A_2_openai_parse_chat_completion_request_media.md</span></a></em></strong>.

## Function Contract And Validation

```python
async def openai_parse_chat_completion_request(
    completion_request: CreateChatCompletionRequest,
    wrap_content: bool,
    settings: Settings,
    max_images_per_request: int | None = None,
    max_videos_per_request: int | None = None,
    allowed_roles: frozenset[str] | None = None,
    preprocessed_image_mask: PreprocessedImageMask | None = None,
) -> _ParsedChatRequest:
    # WALKTHROUGH COMMENT (not in source):
    # The source docstring defines message conversion, media extraction,
    # count limits, a shared byte budget, role validation, and cache-aware
    # image decoding. Tool-call replies must be internally consistent before
    # messages reach a model-specific chat template.
    _validate_tool_message_consistency(completion_request.messages)

    if allowed_roles is not None:
        for m in completion_request.messages:
            role = m.get("role")
            if role not in allowed_roles:
                raise InputError(
                    f"role {role!r} is not supported by this model; "
                    f"allowed roles are {sorted(allowed_roles)}."
                )
```

```text
completion_request.messages
          |
          | _validate_tool_message_consistency(...)
          v
tool-call/reply sequence valid?
          |
      +---+---+
      |       |
     no      yes
      |       |
      v       v
InputError  allowed_roles supplied?
                  |
              +---+---+
              |       |
             no      yes
              |       |
              |       | every message role in allowed_roles?
              |       +-- no --> InputError
              |       +-- yes
              +-------+
                  |
                  v
             parse messages
```

## Message And Content-Part Conversion

```python
    messages: list[TextGenerationRequestMessage] = []
    image_refs: list[MediaRef] = []
    video_refs: list[MediaRef] = []

    for m in completion_request.messages:
        # ``CreateChatCompletionRequest.messages`` carries OpenAI's
        # ``ChatCompletionMessageParam`` TypedDicts (plus a MAX-specific
        # ``video_url`` content part); access via dict keys.
        content = m.get("content")
        raw_tool_calls = m.get("tool_calls")
        tool_calls: list[dict[str, Any]] | None = (
            normalize_tool_call_arguments([dict(tc) for tc in raw_tool_calls])
            if isinstance(raw_tool_calls, list) and raw_tool_calls
            else None
        )
        tool_call_id = m.get("tool_call_id")

        # A client replaying a prior assistant turn echoes back the
        # reasoning under whichever key MAX emitted it.
        reasoning_content_raw = m.get("reasoning_content") or m.get("reasoning")
        reasoning_content = (
            reasoning_content_raw
            if isinstance(reasoning_content_raw, str)
            else None
        )

        if isinstance(content, list):
            message_content: list[MessageContent | dict[str, Any]] = []
            for content_part in content:
                if not isinstance(content_part, dict):
                    raise InputError(
                        "Each entry of message.content must be a content "
                        "part object (e.g. {'type': 'text', 'text': ...}); "
                        f"got {type(content_part).__name__}."
                    )

                part_type = content_part.get("type")
                if part_type == "image_url":
                    image_url = content_part["image_url"]
                    image_refs.append(make_media_ref(image_url["url"]))
                    if wrap_content:
                        message_content.append(
                            ImageContentPart(
                                detail=_coerce_optional_str(
                                    image_url.get("detail")
                                ),
                                max_long_side_pixel=_coerce_positive_int(
                                    image_url.get("max_long_side_pixel")
                                ),
                            )
                        )
                    else:
                        message_content.append(dict(content_part))

                elif part_type == "video_url":
                    video_url = content_part["video_url"]
                    video_refs.append(make_media_ref(video_url["url"]))
                    if wrap_content:
                        message_content.append(
                            VideoContentPart(
                                fps=_coerce_positive_float(
                                    video_url.get("fps")
                                ),
                                max_frames=_coerce_positive_int(
                                    video_url.get("max_frames")
                                ),
                                detail=_coerce_optional_str(
                                    video_url.get("detail")
                                ),
                                max_long_side_pixel=_coerce_positive_int(
                                    video_url.get("max_long_side_pixel")
                                ),
                            )
                        )
                    else:
                        message_content.append(dict(content_part))

                elif part_type == "text":
                    text = content_part.get("text")
                    if text is None:
                        raise InputError(
                            "Content part of type 'text' must include a "
                            "'text' field."
                        )
                    if wrap_content:
                        message_content.append(TextContentPart(text=text))
                    else:
                        message_content.append(dict(content_part))

            messages.append(
                TextGenerationRequestMessage(
                    role=_normalize_openai_role(m["role"]),
                    content=cast(list[MessageContent], message_content),
                    tool_calls=tool_calls,
                    tool_call_id=tool_call_id,
                    reasoning_content=reasoning_content,
                )
            )
        else:
            messages.append(
                TextGenerationRequestMessage(
                    role=_normalize_openai_role(m["role"]),
                    content=content or "",
                    tool_calls=tool_calls,
                    tool_call_id=tool_call_id,
                    reasoning_content=reasoning_content,
                )
            )
```

```text
OpenAI message m
    |
    +-- role --------------------------> _normalize_openai_role(...)
    +-- tool_calls --------------------> normalize_tool_call_arguments(...)
    +-- tool_call_id ------------------> copied
    +-- reasoning/reasoning_content ---> normalized to str | None
    +-- content
          |
          +-- scalar/None --> content or ""
          |
          +-- list
                |
                +-- text ------> TextContentPart or original dict
                +-- image_url -> image_refs + ImageContentPart/original dict
                +-- video_url -> video_refs + VideoContentPart/original dict
                |
                v
       TextGenerationRequestMessage
```

## `wrap_content` Branch

```text
CONTENT PART          wrap_content=True                    wrap_content=False
-------------------   ----------------------------------   --------------------------------
text                  TextContentPart(text=...)            original content-part dict copy
image_url             ImageContentPart(hints...)           original content-part dict copy
video_url             VideoContentPart(hints...)           original content-part dict copy

The media URL is extracted into image_refs/video_refs in both branches.
```

<strong><em><a href="0003_A_2_openai_parse_chat_completion_request_media.md"><span style="color:#0b63ce">Continue to media admission: 0003_A_2_openai_parse_chat_completion_request_media.md</span></a></em></strong>.
