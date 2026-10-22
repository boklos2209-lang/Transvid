"""
Transvid Command Line Interface (CLI)
------------------------------------
Usage:
    python -m src.cli compress <input_file> -o <output.tvid>
    python -m src.cli decompress <input.tvid> -o <output_file>
    python -m src.cli info <input.tvid>
"""

import argparse
import os
import sys
import time

from src.container import TvidReader


def format_size(num_bytes: int) -> str:
    """Formats bytes into human readable format (KB, MB, GB)."""
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

    print(f"[*] Compressing '{input_path}' into '{output_path}'...")
    start_time = time.time()
    orig_size = os.path.getsize(input_path)

    # Determine if audio or video based on extension
    ext = os.path.splitext(input_path)[1].lower()
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]

    if ext in audio_extensions:
        from src.audio import AudioLatentEngine

        engine = AudioLatentEngine(target_bandwidth=args.bandwidth)
        engine.compress_file(input_path, output_path)
    else:
        print(f"Processing video/multimodal media '{input_path}'...")
        # Placeholder for full multimodal pipeline
        from src.audio import AudioLatentEngine

        engine = AudioLatentEngine(target_bandwidth=args.bandwidth)
        engine.compress_file(input_path, output_path)

    comp_size = os.path.getsize(output_path)
    elapsed = time.time() - start_time
    ratio = (orig_size / comp_size) if comp_size > 0 else 0.0
    savings = (1.0 - (comp_size / orig_size)) * 100 if orig_size > 0 else 0.0

    print("\n" + "=" * 50)
    print("           COMPRESSION COMPLETED")
    print("=" * 50)
    print(f" Original Size    : {format_size(orig_size)}")
    print(f" Compressed Size  : {format_size(comp_size)}")
    print(f" Compression Ratio: {ratio:.1f}x smaller")
    print(f" Storage Saved    : {savings:.1f}%")
    print(f" Time Taken       : {elapsed:.2f} seconds")
    print("=" * 50)


def cmd_decompress(args):
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: .tvid file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    reader = TvidReader(input_path)
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + ("_restored.wav" if reader.metadata.has_audio and not reader.metadata.has_video else "_restored.mp4")

    print(f"[*] Decompressing '{input_path}' into '{output_path}'...")
    start_time = time.time()

    if reader.metadata.has_audio and not reader.metadata.has_video:
        from src.audio import AudioLatentEngine

        engine = AudioLatentEngine()
        engine.decompress_file(input_path, output_path)
    else:
        print("Restoring video + audio stream...")
        from src.audio import AudioLatentEngine

        engine = AudioLatentEngine()
        engine.decompress_file(input_path, output_path)

    elapsed = time.time() - start_time
    print(f"[+] Reconstructed successfully in {elapsed:.2f}s -> {output_path}")


def cmd_info(args):
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(input_path)
    reader = TvidReader(input_path)
    meta = reader.metadata

    print("\n" + "=" * 50)
    print(f"       TVID CONTAINER INFO: {os.path.basename(input_path)}")
    print("=" * 50)
    print(f" File Size        : {format_size(file_size)}")
    print(f" Duration         : {meta.duration_seconds:.2f} seconds")
    print(f" Has Audio Stream : {meta.has_audio}")
    if meta.has_audio:
        print(f"   - Audio Codec  : {meta.audio_codec}")
        print(f"   - Sample Rate  : {meta.sample_rate} Hz")
        print(f"   - Channels     : {meta.channels}")
        print(f"   - Codebooks    : {meta.codebook_count}")
        print(f"   - Token Rate   : {meta.tokens_per_second} tokens/sec")
    print(f" Has Video Stream : {meta.has_video}")
    if meta.has_video:
        print(f"   - Video Codec  : {meta.video_codec}")
        print(f"   - Resolution   : {meta.width}x{meta.height}")
        print(f"   - Framerate    : {meta.fps} FPS")
        print(f"   - Total Frames : {meta.total_frames}")
    print(f" Total Chunks     : Video: {len(reader.video_chunks)}, Audio: {len(reader.audio_chunks)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        prog="transvid",
        description="Transvid: Neural Latent Media Compression Machine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compress
    p_comp = subparsers.add_parser("compress", help="Compress media into .tvid format")
    p_comp.add_argument("input", help="Path to input audio/video file")
    p_comp.add_argument("-o", "--output", help="Path to output .tvid file")
    p_comp.add_argument("--bandwidth", type=float, default=6.0, choices=[1.5, 3.0, 6.0, 12.0, 24.0], help="Target audio bandwidth in kbps (default: 6.0)")
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
