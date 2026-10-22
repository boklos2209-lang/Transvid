"""
Unit tests for the .tvid binary container format.
Tests serialization, chunk multiplexing, CRC32 integrity, and parsing.
"""

import os
import unittest
from src.container import StreamMetadata, TvidReader, TvidWriter


class TestTvidContainer(unittest.TestCase):
    def test_container_roundtrip(self):
        # 1. Setup metadata
        meta = StreamMetadata(
            has_video=True,
            video_codec="test-vae-v1",
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=60,
            has_audio=True,
            audio_codec="encodec-24khz",
            sample_rate=24000,
            channels=1,
            codebook_count=8,
            title="Test Sequence",
            duration_seconds=2.0,
        )

        writer = TvidWriter(meta)

        # 2. Add mock video latents (e.g. simulated quantized vectors)
        mock_video_frame_0 = b"\x01\x02\x03\x04" * 1024
        mock_video_frame_1 = b"\x05\x06\x07\x08" * 1024
        writer.add_video_chunk(timestamp_ms=0, raw_latent_bytes=mock_video_frame_0)
        writer.add_video_chunk(timestamp_ms=33, raw_latent_bytes=mock_video_frame_1)

        # 3. Add mock audio tokens (e.g. discrete RVQ codebook indices)
        mock_audio_chunk = b"\x10\x20\x30\x40" * 256
        writer.add_audio_chunk(timestamp_ms=0, raw_token_bytes=mock_audio_chunk)

        # 4. Write to memory buffer
        out_path = "test_output.tvid"
        try:
            writer.write_to_file(out_path)
            self.assertTrue(os.path.exists(out_path))

            # 5. Read back
            reader = TvidReader(out_path)

            # Assert metadata integrity
            self.assertTrue(reader.metadata.has_video)
            self.assertTrue(reader.metadata.has_audio)
            self.assertEqual(reader.metadata.width, 1920)
            self.assertEqual(reader.metadata.height, 1080)
            self.assertEqual(reader.metadata.fps, 30.0)
            self.assertEqual(reader.metadata.sample_rate, 24000)

            # Assert chunks
            self.assertEqual(len(reader.video_chunks), 2)
            self.assertEqual(reader.video_chunks[0][0], 0)
            self.assertEqual(reader.video_chunks[0][1], mock_video_frame_0)
            self.assertEqual(reader.video_chunks[1][0], 33)
            self.assertEqual(reader.video_chunks[1][1], mock_video_frame_1)

            self.assertEqual(len(reader.audio_chunks), 1)
            self.assertEqual(reader.audio_chunks[0][0], 0)
            self.assertEqual(reader.audio_chunks[0][1], mock_audio_chunk)

        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_corrupted_crc_detection(self):
        meta = StreamMetadata(has_video=False, has_audio=False)
        writer = TvidWriter(meta)
        out_path = "corrupt_test.tvid"
        try:
            writer.write_to_file(out_path)
            with open(out_path, "r+b") as f:
                f.seek(10)
                f.write(b"\xFF")  # Corrupt a byte

            with self.assertRaises(ValueError) as ctx:
                TvidReader(out_path)
            self.assertIn("CRC32 mismatch", str(ctx.exception))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
