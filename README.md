# ⚡ Transvid: Neural Latent Media Compression Machine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boklos2209-lang/Transvid/blob/master/notebooks/Transvid_Colab_Demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)

**Transvid** is an extreme neural media compression machine. Instead of saving raw pixel grids or sound waveforms, Transvid encodes media into **compact neural latent representations** and stores them in a custom `.tvid` binary container. On playback, neural decoders reconstruct high-fidelity video and audio without hallucinatory drift.

---

## 🌟 Live Benchmarks

| Media Type | Original Input | Transvid Container (.tvid) | Ratio | Processing Speed | Fidelity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio (Full Track)** | 4.73 MB (`Bruno.mp3`) | **183.4 KB** | **26.4x (96.2% cut)** | ~4.9s | High (EnCodec 24kHz RVQ) |
| **Speech / Acoustic** | 352.8 KB (WAV) | **870 bytes** | **405x reduction** | <0.5s | Crystal clear playback |
| **Video (High Fidelity)** | 1080p / 720p H.264 | `.tvid` (Closed-Loop DPCM) | **3x–6x** | Real-time | ~31+ dB PSNR |
| **Video (Compact Mode)** | 1080p / 720p H.264 | `.tvid` (16x latent pooling) | **10x–15x** | Real-time | Efficient preview / mobile |

---

## 🚀 Key Features

* **Closed-Loop DPCM Prediction:** Eliminates cumulative quantization drift over GOP intervals by synchronizing encoder prediction state with dequantized reconstructions.
* **The `.tvid` Container:** Binary bitstream format with interleaved video/audio chunks, Zstandard/Zlib compression, timestamp synchronization, and CRC32 integrity validation.
* **Extreme Audio Compression:** Reduces audio by **26x–400x** using Residual Vector Quantization (RVQ) down to 1.5–6 kbps.
* **Dual Video Latent Modes:**
  * **Standard Mode (8x TAESD):** Default mode preserving crisp text, sharp edges, and detailed textures.
  * **Compact Mode (16x Pooling):** Opt-in via `--compact` for maximum bandwidth savings.
* **FFmpeg Output Normalization:** Reconstructs back into standardized H.264 / AAC MP4 files without raw frame bloat.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/boklos2209-lang/Transvid.git
cd Transvid

# Install dependencies
pip install -e .
```

---

## 🛠️ Usage

### Compress Media:
```bash
# High-fidelity video compression with quality metric reporting:
transvid compress input.mp4 -o sample.tvid --metrics

# Compact video compression for low bandwidth / mobile:
transvid compress input.mp4 -o sample.tvid --compact

# High-efficiency audio compression (e.g. 6.0 kbps RVQ):
transvid compress song.mp3 -o song.tvid --bandwidth 6.0
```

### Decompress Back to Playable Media:
```bash
# Decompress back to MP4 (with restored synchronized audio):
transvid decompress sample.tvid -o restored.mp4

# Decompress audio:
transvid decompress song.tvid -o restored.mp3
```

### Inspect Container:
```bash
transvid info sample.tvid
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
