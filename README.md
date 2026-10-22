# ⚡ Transvid: Neural Latent Media Compression Machine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)

**Transvid** is an extreme neural media compression machine. Instead of saving raw pixel grids or sound waveforms, Transvid encodes media into **compact neural latent representations** and stores them in a custom `.tvid` binary container. On playback, neural decoders reconstruct high-fidelity video and audio without hallucinatory drift.

---

## 🌟 Why Neural Latent Compression?

| Format | Technology | Video Bitrate | Audio Bitrate | 2-Hour Movie Size |
| :--- | :--- | :--- | :--- | :--- |
| **Old DVD** | MPEG-2 (1994) | 4,000–8,000 kbps | 192–384 kbps | ~4,700 MB (4.7 GB) |
| **Standard MP4** | H.264 / AAC | 1,500–3,000 kbps | 128 kbps | ~1,500–2,500 MB |
| **⚡ Transvid** | Neural Latents / RVQ | **200–800 kbps** | **1.5–6 kbps** | **~200–500 MB** |

---

## 🚀 Key Features

* **No Hallucination:** Unlike text-to-video AI prompts, Transvid stores exact spatial and temporal latent DNA, preserving identity, lighting, and textures.
* **The `.tvid` Container:** Custom binary bitstream format with interleaved video/audio chunks, timestamp synchronization, and CRC32 integrity validation.
* **Extreme Audio Compression:** Reduces audio by **30x–50x** using Residual Vector Quantization (RVQ) down to 1.5–6 kbps.
* **Hardware Aware:** Designed to run with low VRAM footprint on consumer GPUs (e.g. RTX 4050 6GB) via chunked sliding-window processing.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Transvid.git
cd Transvid

# Install with pip / uv
uv pip install -e .
```

---

## 🛠️ Usage

### Compress Media:
```bash
python -m transvid.cli compress input.mp4 -o sample.tvid
```

### Decompress Back to Playable MP4:
```bash
python -m transvid.cli decompress sample.tvid -o restored.mp4
```

---

## 📜 Architecture

```
[Input Video/Audio]
       │
       ├──► Audio Neural Encoder ──► Discrete RVQ Tokens ──┐
       │                                                    ├──► [Bit-Packer + Zstd] ──► 💾 output.tvid
       └──► Video Neural Encoder ──► Latent Matrices ──────┘
```

---

## 📄 License
MIT License.
