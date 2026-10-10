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
