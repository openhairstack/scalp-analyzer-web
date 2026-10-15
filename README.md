---
title: Open Hair Stack — Scalp Analyzer
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Browser-based follicle detection; runs fully locally
---

# Open Hair Stack — Scalp Analyzer (static demo)

**Upload a scalp photo → see follicular units detected and counted, with density stats and an
illustrative graft-planning sandbox.** Everything runs in your browser: the photo is not
uploaded anywhere, and the model is downloaded once from Hugging Face and cached locally.

- Model: [`openhairstack/follicle-fu-v0.1`](https://huggingface.co/openhairstack/follicle-fu-v0.1) (yolo11n fine-tune, ONNX for the web)
- Code: [github.com/openhairstack](https://github.com/openhairstack) · logic in `scalp-analyzer-web/app.js`
- License: Apache 2.0 (code), CC BY 4.0 (data and docs)

## What it does (real) vs. simulates (labeled)

| Feature | Status |
|---|---|
| Follicle-unit detection + hair-count class (`fu1`=1, `fu2`=2, `fu3`=3, `fu4`=4, `fu5`=5+) | real — model runs client-side |
| Counts, mean hairs per unit | real arithmetic on detections |
| FU/cm² density | real when you enter a calibration (pixels per mm) |
| Anterior/posterior "pattern" readout | heuristic — not a Norwood classifier |
| Graft allocation table + chart | simulation — transparent greedy allocation, illustrative only |

## What this is not

Not a medical device. Not diagnostic, not a treatment plan, not FDA/CE cleared. Nothing you do
here should inform any clinical decision. See the disclaimer in the app.

## Why a static Space

Gradio Spaces on organizations need a paid plan. Running the model in the browser means zero
hosting cost forever, works on phones, and — for photos of people's scalps — privacy by
architecture: the image never leaves the device. The Python UI version remains at
[github.com/openhairstack/scalp-analyzer](https://github.com/openhairstack/scalp-analyzer).

## If the model hasn't loaded yet

The banner says so. Training is a one-click Colab notebook
([follicle-detect](https://github.com/openhairstack/follicle-detect)); after training, ONNX
weights publish to the model repo above and this Space activates automatically. The sample
images still demonstrate the full pipeline with placeholder counts.
