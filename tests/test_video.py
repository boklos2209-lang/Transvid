"""
Test suite for Video Latent Engine.
Tests spatio-temporal latent autoencoding, int8 quantization, GOP temporal delta coding,
and full video reconstruction.
"""

import os
import unittest
import cv2
import numpy as np

from src.container import TvidReader
from src.video import VideoLatentEngine


class TestVideoLatentEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_mp4_path = "test_synthetic_input.mp4"
        cls.test_tvid_path = "test_synthetic_output.tvid"
        cls.test_out_mp4_path = "test_synthetic_restored.mp4"

        # Generate 30 frames of synthetic moving video (128x128 @ 30 FPS)
        cls.width = 128
        cls.height = 128
        cls.fps = 30.0
        cls.num_frames = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.test_mp4_path, fourcc, cls.fps, (cls.width, cls.height))

        for i in range(cls.num_frames):
            # Create a dynamic moving circle on a gradient background
            frame = np.zeros((cls.height, cls.width, 3), dtype=np.uint8)
            frame[:, :] = [i * 8 % 255, 100, 150]  # background tint
            center_x = int(cls.width / 2 + 30 * np.cos(i * 0.2))
            center_y = int(cls.height / 2 + 30 * np.sin(i * 0.2))
            cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
            cv2.putText(frame, f"F:{i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            out.write(frame)

        out.release()
        cls.engine = VideoLatentEngine(gop_size=8)

    @classmethod
    def tearDownClass(cls):
        for path in [cls.test_mp4_path, cls.test_tvid_path, cls.test_out_mp4_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_video_compression_and_reconstruction(self):
        orig_size = os.path.getsize(self.test_mp4_path)

        # 1. Compress video to .tvid
        self.engine.compress_video(self.test_mp4_path, self.test_tvid_path)
        self.assertTrue(os.path.exists(self.test_tvid_path))

        comp_size = os.path.getsize(self.test_tvid_path)
        print(f"\n[Video Test] Original MP4: {orig_size} bytes | Compressed TVID: {comp_size} bytes")
        self.assertLess(comp_size, orig_size, "Compressed .tvid must be smaller than original .mp4")
        self.assertIsNotNone(self.engine.last_psnr)
        self.assertTrue(np.isfinite(self.engine.last_psnr))

        # 2. Verify container metadata
        reader = TvidReader(self.test_tvid_path)
        self.assertTrue(reader.metadata.has_video)
        self.assertEqual(reader.metadata.width, self.width)
        self.assertEqual(reader.metadata.height, self.height)
        self.assertEqual(reader.metadata.total_frames, self.num_frames)
        self.assertEqual(len(reader.video_chunks), self.num_frames)

        # 3. Decompress back to MP4
        self.engine.decompress_video(self.test_tvid_path, self.test_out_mp4_path)
        self.assertTrue(os.path.exists(self.test_out_mp4_path))

        # 4. Verify reconstructed video frames
        cap = cv2.VideoCapture(self.test_out_mp4_path)
        self.assertTrue(cap.isOpened())

        frames_read = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.assertEqual(frame.shape, (self.height, self.width, 3))
            frames_read += 1
        cap.release()

        self.assertEqual(frames_read, self.num_frames)
        print(f"[Video Test] Successfully reconstructed all {frames_read} frames.")

    def test_compact_video_and_psnr(self):
        from src.video import compute_psnr

        # Test compute_psnr with identical and noisy images
        img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        self.assertEqual(compute_psnr(img1, img1), float("inf"))

        img2 = np.clip(img1.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        psnr_val = compute_psnr(img1, img2)
        self.assertGreater(psnr_val, 30.0)

        # Test compact 16x mode compression round-trip
        compact_engine = VideoLatentEngine(gop_size=8, compact=True)
        compact_tvid = "test_compact.tvid"
        compact_mp4 = "test_compact_restored.mp4"
        try:
            compact_engine.compress_video(self.test_mp4_path, compact_tvid)
            self.assertTrue(os.path.exists(compact_tvid))
            reader = TvidReader(compact_tvid)
            self.assertIn("compact", reader.metadata.video_codec)

            compact_engine.decompress_video(compact_tvid, compact_mp4)
            self.assertTrue(os.path.exists(compact_mp4))
        finally:
            for p in [compact_tvid, compact_mp4]:
                if os.path.exists(p):
                    os.remove(p)


if __name__ == "__main__":
    unittest.main()
