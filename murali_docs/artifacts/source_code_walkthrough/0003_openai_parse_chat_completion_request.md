# `openai_parse_chat_completion_request(...)`

Source:
[`openai_routes.py`](../../../max/python/max/serve/router/openai_routes.py#L1719)

## One-Line Purpose

`openai_parse_chat_completion_request(...)` converts a validated OpenAI chat
request into MAX's internal message/media pieces.

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
```

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `completion_request` | Validated `CreateChatCompletionRequest` from `_parse_openai_request_body(...)`. |
| Input | `wrap_content` | Whether this tokenizer wants text/media content as typed content parts. |
| Input | `settings` | Server settings, including media byte limits. |
| Input | `max_images_per_request` | Optional model-specific image count limit. |
| Input | `max_videos_per_request` | Optional model-specific video count limit. |
| Input | `allowed_roles` | Optional role allowlist for this model. |
| Input | `preprocessed_image_mask` | Optional tokenizer probe for images already preprocessed. |
| Output | `_ParsedChatRequest` | Normalized messages plus resolved image/video bytes and decoded images. |

```text
CreateChatCompletionRequest
  |
  +-- messages
  +-- image_url content parts
  +-- video_url content parts
  +-- tool_calls / tool_call_id
  +-- reasoning / reasoning_content
  |
  v
openai_parse_chat_completion_request(...)
  |
  v
_ParsedChatRequest(
  messages,
  images,
  videos,
  decoded_images,
)
```

## Output Shape

The function returns `_ParsedChatRequest`:

```python
class _ParsedChatRequest(NamedTuple):
    messages: list[TextGenerationRequestMessage]
    images: list[bytes]
    videos: list[bytes]
    decoded_images: list[Image.Image | None]
```

Visual:

```text
_ParsedChatRequest
  |
  +-- messages
  |     normalized TextGenerationRequestMessage objects
  |
  +-- images
  |     fetched image bytes
  |
  +-- videos
  |     fetched video bytes
  |
  +-- decoded_images
        PIL images, or None when decode was skipped
```

## Worked Example

Assume the validated OpenAI request contains one text message and one image
content part:

```text
INPUT: OpenAI request pieces            ACTION IN openai_parse_chat...             OUTPUT: _ParsedChatRequest pieces
------------------------------------    ----------------------------------------    ------------------------------------
completion_request.messages             validate tool message consistency          ok or InputError
  user content has text + image_url   ------------------------------->

message.role                            validate allowed role                      normalized role
  "user"                              ------------------------------->               "user"

message.content[0]                      normalize text content part                message content part
  {"type":"text",                     ------------------------------->               TextContentPart(
   "text":"What is in this image?"}                                                     "What is in this image?"
                                                                                    )

message.content[1]                      split image placeholder from bytes         two outputs
  {"type":"image_url",                ------------------------------->               message content:
   "image_url":{"url":"data:..."}}                                                        ImageContentPart(...)
                                                                                    image_refs:
                                                                                      [MediaRef("data:...")]

image_refs                              enforce image count limit                  decision
  [MediaRef("data:...")]              ------------------------------->               ok if count <= max_images_per_request

image_refs + settings                   resolve image refs                         request_images
  data URL / remote URL               ------------------------------->               [b"...image bytes..."]

request_images                          decode and validate images                 decoded_images
  [b"...image bytes..."]              ------------------------------->               [PIL.Image(...)]

video_refs                              resolve video refs                         request_videos
  []                                  ------------------------------->               []

all normalized pieces                    package return object                     _ParsedChatRequest
                                      ------------------------------->               messages=[TextGenerationRequestMessage(...)]
                                                                                    images=[b"...image bytes..."]
                                                                                    videos=[]
                                                                                    decoded_images=[PIL.Image(...)]
```

Compact picture:

```text
OpenAI message with image_url
  |
  +-- TextGenerationRequestMessage gets ImageContentPart placeholder
  |
  +-- _ParsedChatRequest.images gets actual image bytes
  |
  +-- _ParsedChatRequest.decoded_images gets decoded PIL image
```

## Sequence View

```text
openai_parse_chat_completion_request(...)
  |
  +-- _validate_tool_message_consistency(...)
  |
  +-- validate each message role, if allowed_roles is supplied
  |
  +-- for each OpenAI message:
  |     |
  |     +-- normalize role
  |     |
  |     +-- normalize tool calls
  |     |
  |     +-- carry reasoning_content / reasoning
  |     |
  |     +-- if content is a list:
  |     |     |
  |     |     +-- text part      -> text content part
  |     |     +-- image_url part -> image placeholder + image ref
  |     |     +-- video_url part -> video placeholder + video ref
  |     |
  |     +-- else:
  |           plain string content
  |
  +-- record image/video count metrics
  |
  +-- enforce image/video count limits
  |
  +-- resolve image refs into bytes
  |
  +-- decode images, unless preprocessing cache says skip
  |
  +-- resolve video refs into bytes
  |
  +-- return _ParsedChatRequest(...)
```

## Message Normalization

OpenAI request message:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What is in this image?"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,...",
        "detail": "high"
      }
    }
  ]
}
```

Normalized MAX message:

```text
TextGenerationRequestMessage
  |
  +-- role = normalized role
  |
  +-- content = [
  |     TextContentPart("What is in this image?"),
  |     ImageContentPart(detail="high"),
  |   ]
  |
  +-- tool_calls = normalized tool calls, if present
  |
  +-- tool_call_id = tool reply id, if present
  |
  +-- reasoning_content = prior reasoning text, if present
```

The image bytes are not stored inside the message content. The message gets an
image placeholder, while the actual bytes are carried separately in
`_ParsedChatRequest.images`.

```text
image_url content part
  |
  +-- message content gets ImageContentPart(...)
  |
  +-- image_refs gets URL/data reference
  |
  +-- later request_images gets fetched bytes
```

## `wrap_content`

`wrap_content` comes from:

```python
tokenizer.expects_content_wrapping
```

It decides whether content parts become typed MAX content objects or remain
plain dictionaries.

```text
wrap_content=True
  |
  +-- text      -> TextContentPart(...)
  +-- image_url -> ImageContentPart(...)
  +-- video_url -> VideoContentPart(...)

wrap_content=False
  |
  +-- keep content part dictionaries
```

## Media Handling

```text
message content
  |
  +-- image_url parts
  |     |
  |     +-- collect image_refs
  |     +-- enforce max_images_per_request
  |     +-- resolve each ref into bytes
  |     +-- decode/validate image bytes
  |     +-- optionally skip pixel decode for cached preprocessed images
  |
  +-- video_url parts
        |
        +-- collect video_refs
        +-- enforce max_videos_per_request
        +-- resolve each ref into bytes
```

All media shares one request-level byte budget from `settings.max_media_bytes`.

```text
images + videos
  -> shared _MediaByteBudget
  -> reject if total fetched/decoded media exceeds the limit
```

## Tool Message Consistency

Before normalizing messages, the function checks tool-call conversations.

```text
assistant message with tool_calls
  |
  +-- function.arguments must be valid JSON
  |
  +-- following tool messages must reference known tool_call_id values
  |
  +-- if conversation continues, every tool_call must be answered
```

Bad tool exchanges raise `InputError`, which becomes a client error response.

## Failure Paths

```text
Invalid tool exchange
  -> InputError

Unsupported role
  -> InputError

Malformed content part
  -> InputError

Too many images/videos
  -> InputError

Media fetch/decode exceeds limits or fails validation
  -> error before request reaches tokenizer/scheduler
```

## Where The Output Goes Next

The route unpacks the returned object:

```python
(
    request_messages,
    request_images,
    request_videos,
    request_decoded_images,
) = await openai_parse_chat_completion_request(...)
```

Then those pieces are used to build `TextGenerationRequest`.

```text
_ParsedChatRequest
  |
  +-- request_messages
  +-- request_images
  +-- request_videos
  +-- request_decoded_images
  |
  v
TextGenerationRequest(...)
  |
  v
tokenizer.new_context(...)
```

Short version:

```text
OpenAI chat messages + media references
  -> validate roles/tools/content parts
  -> normalize messages for MAX
  -> fetch/decode media
  -> return _ParsedChatRequest
```
