# Complete Code Walk: `openai_parse_chat_completion_request(...)` - Media

Source:
[`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L1890)

<strong><em><a href="0003_A_1_openai_parse_chat_completion_request_messages.md"><span style="color:#0b63ce">Return to message parsing: 0003_A_1_openai_parse_chat_completion_request_messages.md</span></a></em></strong>.

## Count Limits Before Download

```python
    if image_refs:
        METRICS.media_items_per_request(len(image_refs), "image")
    if video_refs:
        METRICS.media_items_per_request(len(video_refs), "video")

    # Reject over-limit requests before downloading any media.
    if (
        max_images_per_request is not None
        and len(image_refs) > max_images_per_request
    ):
        METRICS.media_rejections("too_many_images")
        raise InputError(
            f"too many images: {len(image_refs)} exceeds the maximum of "
            f"{max_images_per_request} images per request"
        )
    if (
        max_videos_per_request is not None
        and len(video_refs) > max_videos_per_request
    ):
        METRICS.media_rejections("too_many_videos")
        raise InputError(
            f"too many videos: {len(video_refs)} exceeds the maximum of "
            f"{max_videos_per_request} videos per request"
        )
```

```text
image_refs/video_refs
        |
        | emit request-size metrics
        v
configured count limits exceeded?
        |
    +---+---+
    |       |
   yes      no
    |       |
    v       v
InputError  begin network/media resolution

No image or video bytes are fetched before this count gate passes.
```

## Shared Media Budget And Image Resolution

```python
    # Resolve every reference into bytes under a single per-request media
    # budget shared across all images and videos.
    budget = _request_media_budget(settings)
    resolve_image_tasks = [
        resolve_image_from_url(
            image_url, settings, budget=budget, media_kind="image"
        )
        for image_url in image_refs
    ]
    request_images = await asyncio.gather(*resolve_image_tasks)

    decoded_images: list[Image.Image | None] = []
    if request_images:
        facts: list[_ImageFact] = []
        try:
            with record_ms(METRICS.image_admission_decode_time):
                decoded_images, skip_decode = await asyncio.to_thread(
                    _resolve_and_decode_images,
                    request_images,
                    budget.limit,
                    preprocessed_image_mask,
                    messages,
                    facts,
                )
        except Exception:
            emit_media_facts(facts)
            raise
        emit_media_facts(facts)
        if skip_decode is not None:
            cached = sum(skip_decode)
            METRICS.vision_preprocess_cache_hits(cached)
            METRICS.vision_preprocess_cache_misses(len(skip_decode) - cached)
```

```text
settings.max_media_bytes
          |
          v
shared budget ------------------------------------------------+
          |                                                   |
          | image downloads                                   | video downloads
          v                                                   v
request_images: list[bytes]                           request_videos: list[bytes]
          |
          | asyncio.to_thread(_resolve_and_decode_images, ...)
          v
decoded_images: list[PIL.Image | None]
          |
          +-- PIL.Image -> decoded once and carried to tokenizer
          +-- None ------> preprocessing cache says pixels can be skipped
```

## Video Resolution And Return Object

```python
    resolve_video_tasks = [
        resolve_image_from_url(
            video_url, settings, budget=budget, media_kind="video"
        )
        for video_url in video_refs
    ]
    request_videos = await asyncio.gather(*resolve_video_tasks)

    return _ParsedChatRequest(
        messages, request_images, list(request_videos), decoded_images
    )
```

```text
_ParsedChatRequest
    |
    +-- messages: list[TextGenerationRequestMessage]
    +-- images: list[bytes]
    +-- videos: list[bytes]
    +-- decoded_images: list[PIL.Image | None]
    |
    v
route constructs TextGenerationRequest(...)
```

## Complete Media Input And Output Values

```python
image_refs = [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
]
video_refs = []
max_images_per_request = 4
max_videos_per_request = 1

count_values = {
    "len(image_refs)": 1,
    "len(video_refs)": 0,
    "image_limit_exceeded": False,
    "video_limit_exceeded": False,
}

budget = {
    "limit": 10485760,
    "consumed_before_resolution": 0,
}

resolved_image_bytes_hex = (
    "89504e470d0a1a0a0000000d4948445200000001000000010804000000"
    "b51c0c020000000b4944415478da6364f80f00010501012718e3660000"
    "000049454e44ae426082"
)
resolved_image_bytes_length = 68
request_images = [bytes.fromhex(resolved_image_bytes_hex)]

decoded_images = [
    {
        "type": "PIL.Image.Image",
        "size": (1, 1),
        "format": "PNG",
    }
]
skip_decode = [False]
request_videos = []

parsed_chat_request_output = {
    "messages": [
        {
            "role": "system",
            "content": "You are a concise assistant.",
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning_content": None,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the attached one-pixel image.",
                },
                {
                    "type": "image",
                    "detail": "low",
                    "max_long_side_pixel": 768,
                },
            ],
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning_content": None,
        },
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "id": "call_lookup_001",
                    "type": "function",
                    "function": {
                        "name": "lookup_doc",
                        "arguments": {"query": "one-pixel image"},
                    },
                }
            ],
            "tool_call_id": None,
            "reasoning_content": "The image is intentionally minimal.",
        },
        {
            "role": "tool",
            "content": "A one-pixel image contains exactly one pixel.",
            "tool_calls": None,
            "tool_call_id": "call_lookup_001",
            "reasoning_content": None,
        },
    ],
    "images": request_images,
    "videos": [],
    "decoded_images": decoded_images,
}
```

`resolved_image_bytes_hex` contains all 68 resolved image bytes; no binary
bytes are omitted from this example.
