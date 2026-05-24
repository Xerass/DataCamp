# OCR Insurance Document Classifier

A PyTorch-based multimodal deep learning model that classifies insurance document fields from image scans, combining visual features with document type context.

---

## Project Overview

Insurance documents come in different types (health, auto, etc.) and contain various fields that need to be identified and categorized. This project builds a neural network that looks at a 64×64 grayscale image of a text region and — with the help of a document type signal — predicts whether it is a `primary_id`, `secondary_id`, or another label class.

Despite being called an OCR model, the task is more precisely **document field classification**: the model does not transcribe characters but instead recognizes what *kind* of field the image represents.

---

## Dataset

- **Source**: `ocr_insurance_dataset.pkl` (loaded via `ProjectDataset`)
- **Image size**: 64×64 pixels, grayscale (1 channel)
- **Each sample returns**: `([image_tensor, type_onehot], label)`
  - `image_tensor` — pixel data of the scanned field
  - `type_onehot` — one-hot vector encoding the document type (e.g. health, auto)
  - `label` — integer index of the field class (e.g. `primary_id`, `secondary_id`)
- **Mappings**:
  - `dataset.type_mapping` — maps document type strings to indices
  - `dataset.label_mapping` — maps label strings to indices

---

## Model Architecture — `OCRModel`

The model is **multimodal**: it processes the image and the document type through separate branches, then fuses them for final classification.

```
Image (1, 64, 64)
    │
    ▼
┌─────────────────────────────────┐
│         image_layer             │
│  Conv2d(1→16, 3x3, pad=1)      │
│  ReLU                           │
│  MaxPool2d(2x2)  → (16,32,32)  │
│  Conv2d(16→64, 3x3, pad=1)     │
│  ReLU                           │
│  MaxPool2d(2x2)  → (64,16,16)  │
│  Flatten         → 16384        │
└─────────────────────────────────┘
    │
    ├──────────────────────────────┐
                                   │
Type one-hot (num_types,)          │
    │                              │
    ▼                              │
Linear(num_types → 32)             │
    │                              │
    ▼                              ▼
         torch.cat([img_features, type_features])
                    → (16384 + 32 = 16416)
                         │
                         ▼
                  Linear(16416 → 256)
                         ReLU
                  Linear(256 → num_classes)
                         │
                         ▼
                      Output
```

### Layer Design Decisions

| Component | Choice | Reason |
|---|---|---|
| `Conv2d` channels | 1 → 16 → 64 | Gradually increase feature richness |
| Kernel size | 3×3, padding=1 | Preserves spatial dimensions before pooling |
| `MaxPool2d` | 2×2 twice | Halves spatial size each time, reduces compute |
| `Flatten` inside `image_layer` | Yes | Keeps image branch self-contained as a `Sequential` |
| Type encoding | `nn.Linear` (not `Embedding`) | Type is already one-hot, not a raw integer index |
| Final classifier | Two-layer MLP | Enough capacity to fuse both modalities |

---

## Training

```python
model     = OCRModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    for img_batch, label_batch in loader:
        image      = img_batch[0]   # image tensors
        type_input = img_batch[1]   # one-hot type vectors

        optimizer.zero_grad()
        output = model(image, type_input)
        loss   = criterion(output, label_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10 — Loss: {total_loss/len(loader):.4f}")
```

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=1e-3 |
| Loss function | CrossEntropyLoss |
| Epochs | 10 |
| Batch size | 32 |

---

## Key Learnings

### 1. Multimodal Inputs Require Careful Unpacking
The dataset returns `img` as a **list** `[image_tensor, type_tensor]`, not a stacked tensor. The DataLoader collates these into `img_batch[0]` and `img_batch[1]` — attempting to slice with `img_batch[:, 0]` raises a `TypeError` because list indices don't support tuple indexing.

### 2. One-Hot vs Integer Encoding Changes the Layer Type
When a categorical input is already one-hot encoded, `nn.Embedding` is the wrong tool — it expects a raw integer index. `nn.Linear(num_types, 32)` is the correct layer since it can directly consume a float vector.

### 3. `forward` Must Be a Class Method, Not Nested
Defining `forward` *inside* `__init__` makes it a local function, invisible to PyTorch's module system. This raises `NotImplementedError: missing forward function`. Proper indentation at the class level is essential.

### 4. Print Placement Affects Epoch Logging
Placing the `print` statement inside the batch loop prints loss after every batch rather than every epoch. It must sit one indentation level outside the inner loop to report per-epoch averages correctly.

### 5. Channel Dimensions Must Be Consistent
Changing `out_channels` of one `Conv2d` requires updating the `in_channels` of the next. The final `Linear` input size depends on the last conv's output channels × spatial dimensions: `64 × 16 × 16 = 16384`.

### 6. Task Framing vs Naming
Despite the name "OCR", this model does not perform character transcription. The labels (`primary_id`, `secondary_id`) indicate **field classification** — the model learns what a region *is*, not what it *says*. Understanding the actual task from the data (not just the project title) is critical before designing architecture.

---



## Dependencies

```
torch
torchvision
numpy
matplotlib
pickle
```
