# Section 14.3: Azure Computer Vision

> **Source:** Prosise, Ch. 14 - "The Computer Vision Service"  
> **Prerequisites:** [Section 14.2](./section-02-azure-setup-and-authentication.md) | [Chapter 10](../chapter-10-convolutional-neural-networks/README.md) | [Chapter 12](../chapter-12-object-detection/README.md)  
> **Glossary:** [classification](../../GLOSSARY.md#classification) | [deep-learning](../../GLOSSARY.md#deep-learning)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Vision APIs Without Training a CNN

[Chapter 10](../chapter-10-convolutional-neural-networks/README.md) taught CNN training; [Chapter 12](../chapter-12-object-detection/README.md) added bounding boxes. **Azure Computer Vision** exposes captioning, tagging, detection, OCR, face attributes, and moderation - callable in minutes.

Prosise's **Intellipix** captions uploads and stores keywords for search ("castles," "water in foreground").

> **Humorous analogy:** Training ImageNet from scratch is growing your own vegetables. Computer Vision API is the farmers' market.

> **In plain English:** Send image bytes; get captions, tags, boxes, or text as JSON.

---

## Setup

```python
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os

client = ComputerVisionClient(
    os.environ['AZURE_VISION_ENDPOINT'],
    CognitiveServicesCredentials(os.environ['AZURE_VISION_KEY']),
)
```

---

## Image Captioning

```python
with open('Data/dubai.jpg', 'rb') as f:
    result = client.describe_image_in_stream(f)
    for c in result.captions:
        print(f'{c.text} ({c.confidence:.1%})')
# A man riding a sand dune (53.8%)
```

Confidence $c \in [0,1]$:

$$
\text{rank captions by } c_i \text{ descending}
$$
> **Readable form:** sort captions by confidence from highest to lowest

Use for alt-text, auto-descriptions, search indexing.

---

## Image Tagging

```python
with open('Data/dubai.jpg', 'rb') as f:
    result = client.tag_image_in_stream(f)
    for tag in result.tags:
        print(f'{tag.name} ({tag.confidence:.1%})')
# dune (99.5%), sky (99.2%), desert (98.1%), person (96.1%), ...
```

Store tags in your database for keyword search - different from ImageNet class indices.

---

## Object Detection

```python
from matplotlib.patches import Rectangle

def annotate(name, conf, bbox, ax, min_conf=0.5):
    if conf > min_conf:
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', lw=2))
        ax.text(x + w/2, y, f'{name} ({conf:.0%})', ha='center', color='white',
                fontweight='bold', bbox=dict(facecolor='red'))

with open('Data/xian.jpg', 'rb') as f:
    result = client.detect_objects_in_stream(f)
    for obj in result.objects:
        annotate(obj.object_property, obj.confidence, obj.rectangle, ax)
```

Matches [Section 12.1](../chapter-12-object-detection/section-01-from-classification-to-detection.md) output format. **Custom Vision** ([Chapter 12](../chapter-12-object-detection/section-07-azure-custom-vision.md)) when generic classes fail.

---

## Face Detection

```python
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes

with open('Data/amsterdam.jpg', 'rb') as f:
    result = client.analyze_image_in_stream(f, visual_features=[VisualFeatureTypes.faces])
    for face in result.faces:
        r = face.face_rectangle
        # annotate: f'{face.gender} ({face.age})'
```

> **Note:** Dedicated **Face service** offers recognition and verification - **restricted access** under Responsible AI. Apply via Microsoft if needed.

---

## Adult Content Moderation

```python
with open('Data/maui.jpg', 'rb') as f:
    result = client.analyze_image_in_stream(f, visual_features=[VisualFeatureTypes.adult])
    a = result.adult
    print(f'Raciness: {a.racy_score:.3f}, is_racy: {a.is_racy_content}')
```

Scores on $[0,1]$; booleans use 0.5 threshold. Combine with human review for edge cases.

---

## OCR: Printed Text

```python
with open('Data/1040-es.jpg', 'rb') as f:
    result = client.recognize_printed_text_in_stream(f)
    for region in result.regions:
        for line in region.lines:
            text = ' '.join(w.text for w in line.words)
            print(text)
```

Contrast [Chapter 03 MNIST](../chapter-03-classification-models/section-08-case-study-mnist-digit-recognition.md) - custom digits on clean 28×28 vs API on real documents.

---

## OCR: Handwriting (Read API)

Async - poll until complete:

```python
import time
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes

with open('Data/1040-es.jpg', 'rb') as f:
    response = client.read_in_stream(f, raw=True)
    op_id = response.headers['Operation-Location'].split('/')[-1]
    while True:
        results = client.get_read_result(op_id)
        if results.status != OperationStatusCodes.running:
            break
        time.sleep(1)
    for page in results.analyze_result.read_results:
        for line in page.lines:
            print(line.text)
            # handwriting filter: line.appearance.style.name == 'handwriting'
```

Wrap with timeout and retries ([Section 14.8](./section-08-production-considerations.md)).

---

## 4 MB Image Limit

Oversized images raise `ComputerVisionErrorResponseException`. Resize before upload:

```python
from PIL import Image
import io

def resize_if_needed(path, max_bytes=4_000_000):
    data = open(path, 'rb').read()
    if len(data) <= max_bytes:
        return data
    img = Image.open(io.BytesIO(data))
    img.thumbnail((2000, 2000))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()
```

---

## API vs Custom

| Scenario | API | Custom |
|----------|-----|--------|
| Generic photo tags | ✓ | Overkill |
| Factory defects | Maybe | YOLO / Custom Vision |
| MNIST digits | Overkill | Simple CNN |
| Varied document OCR | ✓ Read API | Hard to match |
| Edge real-time | Latency/cost | TFLite |

---

## Key Takeaways

1. **Computer Vision** - caption, tag, detect, face, moderate, OCR
2. **Confidence scores** on captions and tags
3. **Read API** is async - poll for completion
4. **4 MB limit** - preprocess large images
5. **Face service** restricted - basic attributes via Vision
6. **Custom Vision** for domain-specific detection

---

## Check Your Understanding

1. Difference between `describe_image` and `tag_image`?
2. How does detect output relate to Chapter 12 boxes?
3. Why is `read_in_stream` asynchronous?
4. What happens with a 10 MB image?
5. When use Custom Vision over generic detect?

---

## References

- Prosise, J. *Applied Machine Learning and AI for Engineers* (O'Reilly, 2023), Ch. 14 - Computer Vision
- [Computer Vision docs](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/)
- [Chapter 12 - Custom Vision](../chapter-12-object-detection/section-07-azure-custom-vision.md)

---

**Next:** [Section 14.4 - Azure Language Services](./section-04-azure-language-services.md)



