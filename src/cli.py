"""
Transvid Command Line Interface (CLI)
------------------------------------
Universal CLI for neural latent compression of video, audio, and audiovisual media.

Usage:
    python -m src.cli compress <input_file> [-o <output.tvid>] [--bandwidth 6.0] [--gop 16]
    python -m src.cli decompress <input.tvid> [-o <output_file>]
    python -m src.cli info <input.tvid>
"""

import argparse
import os
import sys
import time

from src.container import TvidReader


def format_size(num_bytes: int) -> str:
    """Formats bytes into human-readable format (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def cmd_compress(args):
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".tvid"

    print(f"[*] Starting Transvid neural compression: '{input_path}' -> '{output_path}'")
    start_time = time.time()
    orig_size = os.path.getsize(input_path)

    ext = os.path.splitext(input_path)[1].lower()
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
    video_extensions = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

    is_video = False
    compact_mode = getattr(args, "compact", False)
    calculate_metrics = getattr(args, "metrics", False)

    if ext in audio_extensions:
        from src.audio import AudioLatentEngine

        engine = AudioLatentEngine(target_bandwidth=args.bandwidth)
        engine.compress_file(input_path, output_path)
    elif ext in video_extensions:
        is_video = True
        from src.video import VideoLatentEngine

        v_engine = VideoLatentEngine(gop_size=args.gop, compact=compact_mode)
        v_engine.compress_video(input_path, output_path)
    else:
        print(f"Warning: Unknown extension '{ext}'. Attempting video compression...", file=sys.stderr)
        is_video = True
        from src.video import VideoLatentEngine

        v_engine = VideoLatentEngine(gop_size=args.gop, compact=compact_mode)
        v_engine.compress_video(input_path, output_path)

    comp_size = os.path.getsize(output_path)
    elapsed = time.time() - start_time
    ratio = (orig_size / comp_size) if comp_size > 0 else 0.0
    savings = (1.0 - (comp_size / orig_size)) * 100 if orig_size > 0 else 0.0

    psnr_score = None
    if calculate_metrics and is_video:
        import tempfile
        import cv2
        from src.video import compute_psnr
        fd_tmp, tmp_rec = tempfile.mkstemp(suffix=".mp4")
        os.close(fd_tmp)
        try:
            v_engine.decompress_video(output_path, tmp_rec)
            cap_orig = cv2.VideoCapture(input_path)
            cap_rec = cv2.VideoCapture(tmp_rec)
            psnrs = []
            while True:
                r1, f1 = cap_orig.read()
                r2, f2 = cap_rec.read()
                if not r1 or not r2:
                    break
                # Resize if minor dimension mismatch
                if f1.shape != f2.shape:
                    f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
                psnrs.append(compute_psnr(f1, f2))
            cap_orig.release()
            cap_rec.release()
            if psnrs:
                psnr_score = float(sum(psnrs) / len(psnrs))
        finally:
            if os.path.exists(tmp_rec):
                os.remove(tmp_rec)

    print("\n" + "=" * 52)
    print("           TRANSVID COMPRESSION REPORT")
    print("=" * 52)
    print(f" Source File      : {os.path.basename(input_path)}")
    print(f" Target File      : {os.path.basename(output_path)}")
    print(f" Original Size    : {format_size(orig_size)}")
    print(f" Compressed Size  : {format_size(comp_size)}")
    print(f" Compression Ratio: {ratio:.1f}x smaller")
    print(f" Storage Reduction: {savings:.1f}%")
    if is_video:
        print(f" Mode             : {'Compact (16x latent pooling)' if compact_mode else 'Standard (8x TAESD crisp)'}")
    if psnr_score is not None:
        print(f" Fidelity (PSNR)  : {psnr_score:.2f} dB")
    print(f" Encoding Speed   : {elapsed:.2f} seconds")
    print("=" * 52)


def cmd_decompress(args):
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: .tvid file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    reader = TvidReader(input_path)
    meta = reader.metadata

    if not output_path:
        base, _ = os.path.splitext(input_path)
        if meta.has_video:
            output_path = base + "_restored.mp4"
        elif meta.has_audio:
            output_path = base + "_restored.wav"
        else:
            output_path = base + "_restored.bin"

    print(f"[*] Decompressing '{input_path}' -> '{output_path}'...")
    start_time = time.time()

    if meta.has_video:
        from src.video import VideoLatentEngine

        v_engine = VideoLatentEngine()
        v_engine.decompress_video(input_path, output_path)
    elif meta.has_audio:
        from src.audio import AudioLatentEngine

        a_engine = AudioLatentEngine()
        a_engine.decompress_file(input_path, output_path)
    else:
        raise ValueError("Container has neither audio nor video streams.")

    elapsed = time.time() - start_time
    print(f"[+] Reconstruction complete in {elapsed:.2f}s -> {output_path}")


def cmd_info(args):
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(input_path)
    reader = TvidReader(input_path)
    meta = reader.metadata

    print("\n" + "=" * 52)
    print(f"       TVID CONTAINER DETAILS: {os.path.basename(input_path)}")
    print("=" * 52)
    print(f" Container Size   : {format_size(file_size)}")
    print(f" Duration         : {meta.duration_seconds:.2f} seconds")
    print(f" Audio Stream     : {'Active' if meta.has_audio else 'None'}")
    if meta.has_audio:
        print(f"   - Codec        : {meta.audio_codec}")
        print(f"   - Sample Rate  : {meta.sample_rate} Hz")
        print(f"   - Channels     : {meta.channels}")
        print(f"   - Codebooks    : {meta.codebook_count}")
        print(f"   - Token Rate   : {meta.tokens_per_second} tokens/sec")
    print(f" Video Stream     : {'Active' if meta.has_video else 'None'}")
    if meta.has_video:
        print(f"   - Codec        : {meta.video_codec}")
        print(f"   - Resolution   : {meta.width}x{meta.height}")
        print(f"   - Framerate    : {meta.fps} FPS")
        print(f"   - Total Frames : {meta.total_frames}")
        print(f"   - Latent Shape : {meta.latent_shape}")
    print(f" Stream Chunks    : Video: {len(reader.video_chunks)} chunks, Audio: {len(reader.audio_chunks)} chunks")
    print("=" * 52)


def main():
    parser = argparse.ArgumentParser(
        prog="transvid",
        description="Transvid: Neural Latent Media Compression Machine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compress
    p_comp = subparsers.add_parser("compress", help="Compress audio or video into .tvid format")
    p_comp.add_argument("input", help="Path to input audio/video file")
    p_comp.add_argument("-o", "--output", help="Path to output .tvid file")
    p_comp.add_argument("--bandwidth", type=float, default=6.0, choices=[1.5, 3.0, 6.0, 12.0, 24.0], help="Target audio bandwidth in kbps (default: 6.0)")
    p_comp.add_argument("--gop", type=int, default=16, help="Video Group of Pictures (GOP) size for keyframing (default: 16)")
    p_comp.add_argument("--compact", action="store_true", help="Enable 16x compact latent spatial pooling for extreme compression")
    p_comp.add_argument("--metrics", action="store_true", help="Compute PSNR fidelity metric after compression")
    p_comp.set_defaults(func=cmd_compress)

    # Decompress
    p_decomp = subparsers.add_parser("decompress", help="Decompress .tvid back into playable media")
    p_decomp.add_argument("input", help="Path to input .tvid file")
    p_decomp.add_argument("-o", "--output", help="Path to output restored file")
    p_decomp.set_defaults(func=cmd_decompress)

    # Info
    p_info = subparsers.add_parser("info", help="Inspect .tvid metadata and streams")
    p_info.add_argument("input", help="Path to .tvid file")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
