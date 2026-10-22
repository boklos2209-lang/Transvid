"""
Video Latent Engine for Transvid
--------------------------------
Compresses video frames into compact spatio-temporal latent representations
using pure PyTorch TAESD (Tiny AutoEncoder for Stable Diffusion) with
pretrained weights, int8 quantization, and temporal delta coding.

Zero heavy dependencies (no diffusers/torchao required).
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.hub import download_url_to_file

from src.container import StreamMetadata, TvidReader, TvidWriter

TAESD_ENCODER_URL = "https://github.com/madebyollin/taesd/raw/main/taesd_encoder.pth"
TAESD_DECODER_URL = "https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth"


def conv(n_in: int, n_out: int, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class ResBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def build_encoder(latent_channels: int = 4) -> nn.Sequential:
    return nn.Sequential(
        conv(3, 64),
        ResBlock(64, 64),
        conv(64, 64, stride=2, bias=False),  # 2x down
        ResBlock(64, 64),
        ResBlock(64, 64),
        ResBlock(64, 64),
        conv(64, 64, stride=2, bias=False),  # 4x down
        ResBlock(64, 64),
        ResBlock(64, 64),
        ResBlock(64, 64),
        conv(64, 64, stride=2, bias=False),  # 8x down
        ResBlock(64, 64),
        ResBlock(64, 64),
        ResBlock(64, 64),
        conv(64, latent_channels),
    )


def build_decoder(latent_channels: int = 4) -> nn.Sequential:
    return nn.Sequential(
        Clamp(),
        conv(latent_channels, 64),
        nn.ReLU(),
        ResBlock(64, 64),
        ResBlock(64, 64),
        ResBlock(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        ResBlock(64, 64),
        ResBlock(64, 64),
        ResBlock(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        ResBlock(64, 64),
        ResBlock(64, 64),
        ResBlock(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        ResBlock(64, 64),
        conv(64, 3),
    )


class VideoLatentEngine:
    """
    Manages video frame extraction, neural latent encoding via pure PyTorch TAESD,
    and photorealistic neural reconstruction.
    """

    def __init__(self, device: Optional[str] = None, gop_size: int = 16):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.gop_size = gop_size

        # Build models
        self.encoder = build_encoder(latent_channels=4).to(self.device)
        self.decoder = build_decoder(latent_channels=4).to(self.device)

        self._load_pretrained_weights()

        self.encoder.eval()
        self.decoder.eval()

    def _load_pretrained_weights(self):
        cache_dir = Path.home() / ".cache" / "transvid"
        cache_dir.mkdir(parents=True, exist_ok=True)

        enc_path = cache_dir / "taesd_encoder.pth"
        dec_path = cache_dir / "taesd_decoder.pth"

        if not enc_path.exists():
            print("[*] Downloading TAESD pretrained video encoder weights (4.7 MB)...")
            download_url_to_file(TAESD_ENCODER_URL, str(enc_path))

        if not dec_path.exists():
            print("[*] Downloading TAESD pretrained video decoder weights (4.7 MB)...")
            download_url_to_file(TAESD_DECODER_URL, str(dec_path))

        self.encoder.load_state_dict(torch.load(str(enc_path), map_location=self.device, weights_only=True))
        self.decoder.load_state_dict(torch.load(str(dec_path), map_location=self.device, weights_only=True))

    def _quantize_latent(self, latent: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantizes continuous float32/float16 latent to int8."""
        max_val = np.max(np.abs(latent))
        scale = max(float(max_val) / 127.0, 1e-7)
        quantized = np.clip(np.round(latent / scale), -128, 127).astype(np.int8)
        return quantized, scale

    def _dequantize_latent(self, quantized: np.ndarray, scale: float) -> np.ndarray:
        """Dequantizes int8 latent back to float32."""
        return (quantized.astype(np.float32) * scale).astype(np.float32)

    def compress_video(self, input_video_path: str, output_tvid_path: str):
        """Encodes an MP4/video file into .tvid format using spatio-temporal latents."""
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video '{input_video_path}'")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        # Ensure width and height are divisible by 8 for 8x downsampling
        pad_w = (8 - (width % 8)) % 8
        pad_h = (8 - (height % 8)) % 8
        target_w = width + pad_w
        target_h = height + pad_h

        latent_shape = (4, target_h // 8, target_w // 8)

        meta = StreamMetadata(
            has_video=True,
            video_codec="taesd-pure-v1",
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            latent_shape=latent_shape,
            duration_seconds=(total_frames / fps) if fps > 0 else 0.0,
        )

        writer = TvidWriter(meta)

        frame_idx = 0
        prev_latent: Optional[np.ndarray] = None

        print(f"[*] Encoding video frames ({width}x{height} @ {fps:.1f} FPS, total: {total_frames})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if pad_w > 0 or pad_h > 0:
                frame_rgb = cv2.copyMakeBorder(
                    frame_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

            # Normalize to [0, 1] and convert to tensor [1, 3, H, W]
            frame_tensor = (
                torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(self.device)

            with torch.inference_mode():
                latent_tensor = self.encoder(frame_tensor)
                latent = latent_tensor.squeeze(0).cpu().numpy()  # [4, H/8, W/8]

            timestamp_ms = int((frame_idx / fps) * 1000)
            is_keyframe = (frame_idx % self.gop_size == 0) or (prev_latent is None)

            if is_keyframe:
                # Keyframe (I-Frame): store full latent
                target_latent = latent
                frame_type = 0
            else:
                # Delta frame (P-Frame): store temporal difference
                target_latent = latent - prev_latent
                frame_type = 1

            prev_latent = latent

            quantized, scale = self._quantize_latent(target_latent)

            # Header per chunk: [frame_type (uint8), scale (float32)] + raw int8 payload
            scale_bytes = np.float32(scale).tobytes()
            type_byte = bytes([frame_type])
            chunk_payload = type_byte + scale_bytes + quantized.tobytes()

            writer.add_video_chunk(timestamp_ms=timestamp_ms, raw_latent_bytes=chunk_payload, compress=True)
            frame_idx += 1

        cap.release()
        writer.write_to_file(output_tvid_path)
        print(f"[+] Compressed {frame_idx} video frames into '{output_tvid_path}'.")

    def decompress_video(self, input_tvid_path: str, output_video_path: str):
        """Reconstructs full-resolution video from a .tvid container."""
        reader = TvidReader(input_tvid_path)
        if not reader.metadata.has_video or not reader.video_chunks:
            raise ValueError(f"No video stream found in {input_tvid_path}")

        meta = reader.metadata
        width, height = meta.width, meta.height
        fps = meta.fps
        latent_shape = tuple(meta.latent_shape)

        # Initialize video writer with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        prev_latent: Optional[np.ndarray] = None
        print(f"[*] Decompressing {len(reader.video_chunks)} video frames into '{output_video_path}'...")

        for chunk_idx, (timestamp_ms, raw_chunk) in enumerate(reader.video_chunks):
            frame_type = raw_chunk[0]
            scale = np.frombuffer(raw_chunk[1:5], dtype=np.float32)[0]
            quantized_bytes = raw_chunk[5:]

            quantized = np.frombuffer(quantized_bytes, dtype=np.int8).reshape(latent_shape)
            latent = self._dequantize_latent(quantized, scale)

            if frame_type == 0:  # Keyframe
                curr_latent = latent
            else:  # Delta frame
                curr_latent = prev_latent + latent

            prev_latent = curr_latent

            # Pass through pure PyTorch neural decoder
            latent_tensor = torch.from_numpy(curr_latent).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                rec_tensor = self.decoder(latent_tensor)

            # Convert from [0, 1] tensor back to [0, 255] uint8 RGB image
            rec_frame = (
                (rec_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255.0)
                .astype(np.uint8)
            )

            # Crop padding if dimensions were padded
            rec_frame = rec_frame[:height, :width]

            # Convert RGB to BGR for OpenCV writer
            rec_bgr = cv2.cvtColor(rec_frame, cv2.COLOR_RGB2BGR)
            out.write(rec_bgr)

        out.release()
        print(f"[+] Reconstructed video written to '{output_video_path}'.")
