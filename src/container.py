"""
Transvid Container Specification (.tvid)
----------------------------------------
A binary container format designed specifically for streaming and storing
neural latent representations (video latents + audio tokens).

Binary Layout:
+-------------------+-----------------+----------------------------------------+
| Field             | Type            | Description                            |
+-------------------+-----------------+----------------------------------------+
| Magic             | char[4]         | 'TVID' (0x54, 0x56, 0x49, 0x44)        |
| Version           | uint16          | Format version (1)                     |
| Flags             | uint16          | Bit 0: Has Video, Bit 1: Has Audio     |
| Metadata Length   | uint32          | Length of JSON metadata block in bytes |
| Metadata JSON     | char[len]       | UTF-8 JSON (codec, dims, fps, rate)    |
| Chunk Count       | uint32          | Total number of data chunks            |
+-------------------+-----------------+----------------------------------------+
| Chunks...                                                                    |
|   - Stream ID     | uint8           | 0 = Video, 1 = Audio                   |
|   - Timestamp Ms  | uint32          | Presentation timestamp in milliseconds |
|   - Payload Len   | uint32          | Length of compressed payload           |
|   - Payload Bytes | uint8[len]      | Compressed latent data                 |
+-------------------+-----------------+----------------------------------------+
| Checksum          | uint32          | CRC32 checksum over entire file        |
+-------------------+-----------------+----------------------------------------+
"""

import io
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from typing import BinaryIO, Dict, List, Optional, Tuple

MAGIC = b"TVID"
CURRENT_VERSION = 1

FLAG_HAS_VIDEO = 1 << 0
FLAG_HAS_AUDIO = 1 << 1


@dataclass
class StreamMetadata:
    # Video metadata
    has_video: bool = False
    video_codec: str = "none"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    latent_shape: Tuple[int, ...] = ()

    # Audio metadata
    has_audio: bool = False
    audio_codec: str = "none"
    sample_rate: int = 0
    channels: int = 0
    codebook_count: int = 0
    tokens_per_second: int = 0

    # General info
    title: str = ""
    duration_seconds: float = 0.0


@dataclass
class Chunk:
    stream_id: int  # 0 = Video, 1 = Audio
    timestamp_ms: int
    data: bytes


class TvidWriter:
    """Writes a .tvid binary container."""

    def __init__(self, metadata: StreamMetadata):
        self.metadata = metadata
        self.chunks: List[Chunk] = []

    def add_video_chunk(self, timestamp_ms: int, raw_latent_bytes: bytes, compress: bool = True):
        payload = zlib.compress(raw_latent_bytes, level=6) if compress else raw_latent_bytes
        self.chunks.append(Chunk(stream_id=0, timestamp_ms=timestamp_ms, data=payload))

    def add_audio_chunk(self, timestamp_ms: int, raw_token_bytes: bytes, compress: bool = True):
        payload = zlib.compress(raw_token_bytes, level=6) if compress else raw_token_bytes
        self.chunks.append(Chunk(stream_id=1, timestamp_ms=timestamp_ms, data=payload))

    def write_to_file(self, file_path: str):
        with open(file_path, "wb") as f:
            self.write(f)

    def write(self, stream: BinaryIO):
        buffer = io.BytesIO()

        # 1. Magic
        buffer.write(MAGIC)

        # 2. Version
        buffer.write(struct.pack(">H", CURRENT_VERSION))

        # 3. Flags
        flags = 0
        if self.metadata.has_video:
            flags |= FLAG_HAS_VIDEO
        if self.metadata.has_audio:
            flags |= FLAG_HAS_AUDIO
        buffer.write(struct.pack(">H", flags))

        # 4. Metadata JSON
        meta_dict = asdict(self.metadata)
        meta_bytes = json.dumps(meta_dict).encode("utf-8")
        buffer.write(struct.pack(">I", len(meta_bytes)))
        buffer.write(meta_bytes)

        # 5. Chunk Count
        buffer.write(struct.pack(">I", len(self.chunks)))

        # 6. Chunks
        for chunk in self.chunks:
            buffer.write(struct.pack(">BII", chunk.stream_id, chunk.timestamp_ms, len(chunk.data)))
            buffer.write(chunk.data)

        # 7. Compute CRC32
        data_so_far = buffer.getvalue()
        crc = zlib.crc32(data_so_far)
        buffer.write(struct.pack(">I", crc))

        # Flush to target stream
        stream.write(buffer.getvalue())


class TvidReader:
    """Reads and parses a .tvid binary container."""

    def __init__(self, file_path_or_bytes: bytes | str):
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "rb") as f:
                self.raw_data = f.read()
        else:
            self.raw_data = file_path_or_bytes

        self.metadata: StreamMetadata = StreamMetadata()
        self.video_chunks: List[Tuple[int, bytes]] = []  # (timestamp_ms, decompressed_bytes)
        self.audio_chunks: List[Tuple[int, bytes]] = []

        self._parse()

    def _parse(self):
        stream = io.BytesIO(self.raw_data)

        # Check size for header + CRC
        if len(self.raw_data) < 16:
            raise ValueError("File is too small to be a valid .tvid container")

        # Verify CRC32
        expected_crc = struct.unpack(">I", self.raw_data[-4:])[0]
        actual_crc = zlib.crc32(self.raw_data[:-4])
        if expected_crc != actual_crc:
            raise ValueError(f"CRC32 mismatch! Corrupt .tvid file (expected {expected_crc}, got {actual_crc})")

        # 1. Magic
        magic = stream.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic bytes: expected {MAGIC!r}, got {magic!r}")

        # 2. Version
        (version,) = struct.unpack(">H", stream.read(2))
        if version > CURRENT_VERSION:
            raise ValueError(f"Unsupported .tvid version: {version}")

        # 3. Flags
        (flags,) = struct.unpack(">H", stream.read(2))

        # 4. Metadata JSON
        (meta_len,) = struct.unpack(">I", stream.read(4))
        meta_bytes = stream.read(meta_len)
        meta_dict = json.loads(meta_bytes.decode("utf-8"))
        self.metadata = StreamMetadata(**meta_dict)

        # 5. Chunk Count
        (chunk_count,) = struct.unpack(">I", stream.read(4))

        # 6. Read Chunks
        for _ in range(chunk_count):
            stream_id, timestamp_ms, payload_len = struct.unpack(">BII", stream.read(9))
            compressed_payload = stream.read(payload_len)
            decompressed_payload = zlib.decompress(compressed_payload)

            if stream_id == 0:
                self.video_chunks.append((timestamp_ms, decompressed_payload))
            elif stream_id == 1:
                self.audio_chunks.append((timestamp_ms, decompressed_payload))
