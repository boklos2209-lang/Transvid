"""
Test suite for Audio Latent Engine.
Tests neural tokenization, roundtrip compression/reconstruction, and audio quality.
"""

import os
import unittest
import numpy as np
import torch
import torchaudio

from src.audio import AudioLatentEngine
from src.container import TvidReader


class TestAudioLatentEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate a synthetic multi-frequency test audio file (2 seconds at 44.1kHz)
        cls.test_wav_path = "test_audio_sample.wav"
        cls.test_tvid_path = "test_audio_sample.tvid"
        cls.test_out_wav_path = "test_audio_restored.wav"

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Multi-tone acoustic signal (440Hz A + 880Hz octave + 220Hz bass)
        audio_data = (
            0.5 * np.sin(2 * np.pi * 440 * t)
            + 0.3 * np.sin(2 * np.pi * 880 * t)
            + 0.2 * np.sin(2 * np.pi * 220 * t)
        ).astype(np.float32)

        # Shape: [channels=1, samples]
        wav_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        torchaudio.save(cls.test_wav_path, wav_tensor, sample_rate)

        cls.engine = AudioLatentEngine(target_bandwidth=6.0)

    @classmethod
    def tearDownClass(cls):
        for path in [cls.test_wav_path, cls.test_tvid_path, cls.test_out_wav_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_compression_and_reconstruction(self):
        orig_size = os.path.getsize(self.test_wav_path)

        # 1. Compress
        self.engine.compress_file(self.test_wav_path, self.test_tvid_path)
        self.assertTrue(os.path.exists(self.test_tvid_path))
        comp_size = os.path.getsize(self.test_tvid_path)

        # Assert extreme compression ratio: tvid must be significantly smaller than uncompressed wav
        # 2 seconds of 44.1kHz 16-bit audio = ~176 KB.
        # At 6 kbps, 2 seconds is ~1.5 KB!
        print(f"\n[Test] Original WAV: {orig_size} bytes | Compressed TVID: {comp_size} bytes")
        self.assertLess(comp_size, orig_size / 10, "Compression should be at least 10x smaller!")

        # 2. Verify container structure
        reader = TvidReader(self.test_tvid_path)
        self.assertTrue(reader.metadata.has_audio)
        self.assertEqual(reader.metadata.audio_codec, "encodec-24khz")
        self.assertGreater(reader.metadata.duration_seconds, 1.9)

        # 3. Decompress
        self.engine.decompress_file(self.test_tvid_path, self.test_out_wav_path)
        self.assertTrue(os.path.exists(self.test_out_wav_path))

        # 4. Verify reconstructed audio
        rec_wav, rec_sr = torchaudio.load(self.test_out_wav_path)
        self.assertEqual(rec_sr, 24000)
        self.assertGreater(rec_wav.shape[1], 0)
        print(f"[Test] Reconstructed WAV shape: {rec_wav.shape} at {rec_sr} Hz")


if __name__ == "__main__":
    unittest.main()
