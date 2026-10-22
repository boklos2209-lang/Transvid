"""
Audio Latent Engine for Transvid
--------------------------------
Compresses audio waveforms into discrete neural tokens using Residual Vector
Quantization (RVQ) and reconstructs audio via neural vocoders.
"""

import io
import math
from typing import BinaryIO, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

from src.container import StreamMetadata, TvidReader, TvidWriter


class AudioLatentEngine:
    """Neural Audio Codec Engine powered by EnCodec."""

    def __init__(self, device: Optional[str] = None, target_bandwidth: float = 6.0):
        """
        target_bandwidth options: 1.5, 3.0, 6.0, 12.0, 24.0 kbps.
        6.0 kbps gives near CD-quality with massive compression.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.target_bandwidth = target_bandwidth
        self.model: Optional[EncodecModel] = None

    def _ensure_model_loaded(self):
        if self.model is None:
            # Instantiate 24kHz model
            self.model = EncodecModel.encodec_model_24khz()
            self.model.set_target_bandwidth(self.target_bandwidth)
            self.model.to(self.device)
            self.model.eval()

    def encode_waveform(
        self, wav: torch.Tensor, sample_rate: int
    ) -> Tuple[np.ndarray, int, float]:
        """
        Encodes a waveform tensor into discrete tokens.
        wav shape: [channels, time] or [1, channels, time]
        Returns:
            tokens: np.ndarray of shape [num_codebooks, time_steps] (uint16)
            original_sample_rate: int
            duration_seconds: float
        """
        self._ensure_model_loaded()

        if wav.dim() == 2:
            wav = wav.unsqueeze(0)  # [1, channels, time]

        original_channels = wav.shape[1]
        original_frames = wav.shape[2]
        duration_seconds = original_frames / sample_rate

        # Resample and convert to mono 24kHz required by EnCodec 24kHz model
        wav_24k = convert_audio(
            wav,
            sr=sample_rate,
            target_sr=self.model.sample_rate,
            target_channels=self.model.channels,
        ).to(self.device)

        with torch.inference_mode():
            # frames is a list of tuples: [(codes, scale)]
            # codes shape: [batch, num_codebooks, time_steps]
            encoded_frames = self.model.encode(wav_24k)

        # Concatenate codes across chunks if audio was split
        codes_list = [frame[0] for frame in encoded_frames]
        all_codes = torch.cat(codes_list, dim=-1)  # [batch, num_codebooks, total_time_steps]

        # Extract first batch and cast to uint16 (codebook indices are 0-1023)
        tokens = all_codes.squeeze(0).cpu().numpy().astype(np.uint16)
        return tokens, sample_rate, duration_seconds

    def decode_tokens(self, tokens: np.ndarray) -> Tuple[torch.Tensor, int]:
        """
        Reconstructs audio waveform from discrete tokens.
        tokens: np.ndarray [num_codebooks, time_steps] (uint16)
        Returns:
            wav: torch.Tensor [1, time]
            sample_rate: int (24000)
        """
        self._ensure_model_loaded()

        # Convert back to torch tensor [batch, num_codebooks, time_steps]
        tokens_tensor = torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0).to(self.device)

        # Structure as expected by Encodec: [(codes, scale)]
        encoded_frames = [(tokens_tensor, None)]

        with torch.inference_mode():
            decoded_wav = self.model.decode(encoded_frames)

        # decoded_wav shape: [1, 1, time] -> squeeze batch
        return decoded_wav.squeeze(0).cpu(), self.model.sample_rate

    def compress_file(self, input_audio_path: str, output_tvid_path: str):
        """Compresses any audio file into a .tvid container."""
        wav, sr = torchaudio.load(input_audio_path)
        tokens, original_sr, duration = self.encode_waveform(wav, sr)

        num_codebooks, time_steps = tokens.shape

        meta = StreamMetadata(
            has_audio=True,
            audio_codec="encodec-24khz",
            sample_rate=sr,
            channels=wav.shape[0],
            codebook_count=num_codebooks,
            tokens_per_second=int(time_steps / max(duration, 0.001)),
            duration_seconds=duration,
        )

        writer = TvidWriter(meta)
        # Store tokens shape and raw bytes
        # Header shape prefix (2x uint32: num_codebooks, time_steps) + uint16 payload
        shape_header = np.array([num_codebooks, time_steps], dtype=np.uint32).tobytes()
        payload = shape_header + tokens.tobytes()

        writer.add_audio_chunk(timestamp_ms=0, raw_token_bytes=payload, compress=True)
        writer.write_to_file(output_tvid_path)

    def decompress_file(self, input_tvid_path: str, output_audio_path: str):
        """Extracts audio tokens from .tvid and reconstructs original audio file."""
        reader = TvidReader(input_tvid_path)
        if not reader.metadata.has_audio or not reader.audio_chunks:
            raise ValueError(f"No audio stream found in {input_tvid_path}")

        # Combine all audio chunks
        full_payload = reader.audio_chunks[0][1]
        num_codebooks, time_steps = np.frombuffer(full_payload[:8], dtype=np.uint32)
        raw_tokens_bytes = full_payload[8:]

        tokens = np.frombuffer(raw_tokens_bytes, dtype=np.uint16).reshape((num_codebooks, time_steps))
        reconstructed_wav, model_sr = self.decode_tokens(tokens)

        # If original was a different sample rate, we can optionally resample or save directly
        torchaudio.save(output_audio_path, reconstructed_wav, model_sr)
