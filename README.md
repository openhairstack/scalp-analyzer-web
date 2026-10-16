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
- Blueprint & Investment: [Commons, Governance & Investment Blueprint](https://github.com/openhairstack/governance/blob/main/COMMONS-AND-INVESTMENT.md)
- Collaboration Intake: [Tally Collaboration Form](https://tally.so/r/o96OkV)
- Project Lead: Salik Shah ([@salik on X](https://x.com/salik))

> **For Clinics, Researchers & Sponsors:** We are partnering with trichologists, physical AI roboticists, and strategic sponsors to advance open biomedical vision and autonomous robotic actuation. Inquiries: [@salik on X](https://x.com/salik).

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

## Client-Side ONNX Model

The detector runs [`openhairstack/follicle-fu-v0.1`](https://huggingface.co/openhairstack/follicle-fu-v0.1) (`model.onnx`, ~10 MB) directly in the browser via `onnxruntime-web` WebAssembly (WASM).
- **100% client-side:** no photo or data ever leaves your machine.
- **Zero server dependency:** once cached by your browser, it runs completely local.
- **Interactive tuning:** adjust the confidence slider (0.01 – 0.50) to explore candidate follicular unit detections in real-time.

---

### Contact & Technical Discussions

A project by Salik Shah:
* **Contact on X:** [@salik](https://x.com/salik)
* **Discussions:** [GitHub Discussions](https://github.com/orgs/openhairstack/discussions)
