# Graph Report - Transvid  (2026-10-22)

## Corpus Check
- 12 files · ~4,994 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 3 file(s) not represented in the graph (top: (none) 2, .ipynb 1)

## Summary
- 142 nodes · 314 edges · 12 communities (5 shown, 7 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `52df1f3f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- container.py
- StreamMetadata
- test_container.py
- VideoLatentEngine
- cli.py
- transvid
- container.py
- ⚡ Transvid: Neural Latent Media Compression Machine
- transvid/__init__.py

## God Nodes (most connected - your core abstractions)
1. `TvidReader` - 26 edges
2. `AudioLatentEngine` - 22 edges
3. `VideoLatentEngine` - 21 edges
4. `TvidWriter` - 19 edges
5. `StreamMetadata` - 16 edges
6. `cmd_compress()` - 12 edges
7. `pack_audio_tokens()` - 11 edges
8. `unpack_audio_tokens()` - 11 edges
9. `cmd_decompress()` - 9 edges
10. `compute_psnr()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `TestTvidContainer` --uses--> `TvidReader`  [INFERRED]
  tests/test_container.py → src/container.py
- `TestVideoLatentEngine` --uses--> `TvidReader`  [INFERRED]
  tests/test_video.py → src/container.py
- `TestAudioLatentEngine` --uses--> `AudioLatentEngine`  [INFERRED]
  tests/test_audio.py → src/audio.py
- `TestTvidContainer` --uses--> `StreamMetadata`  [INFERRED]
  tests/test_container.py → src/container.py
- `TestTvidContainer` --uses--> `TvidWriter`  [INFERRED]
  tests/test_container.py → src/container.py

## Import Cycles
- None detected.

## Communities (12 total, 7 thin omitted)

### Community 1 - "container.py"
Cohesion: 0.25
Nodes (4): Chunk, StreamMetadata, TvidWriter, TestTvidContainer

### Community 2 - "StreamMetadata"
Cohesion: 0.14
Nodes (3): AudioLatentEngine, TvidReader, TestAudioLatentEngine

### Community 3 - "test_container.py"
Cohesion: 0.16
Nodes (5): build_decoder(), build_encoder(), Clamp, conv(), ResBlock

### Community 5 - "cli.py"
Cohesion: 0.24
Nodes (6): cmd_compress(), cmd_decompress(), cmd_info(), format_size(), main(), compute_psnr()

### Community 9 - "⚡ Transvid: Neural Latent Media Compression Machine"
Cohesion: 0.18
Nodes (10): 📜 Architecture, Compress Media:, Decompress Back to Playable Media:, Inspect Container:, 📦 Installation, 🚀 Key Features, 📄 License, 🌟 Live Benchmarks (+2 more)

## Knowledge Gaps
- **1 isolated node(s):** `transvid`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 59 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AudioLatentEngine` connect `StreamMetadata` to `audio.py`, `container.py`, `test_container.py`, `VideoLatentEngine`, `cli.py`, `transvid/__init__.py`?**
  _High betweenness centrality (0.147) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `TvidReader` (e.g. with `AudioLatentEngine` and `VideoLatentEngine`) actually correct?**
  _`TvidReader` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transvid` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `StreamMetadata` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._
- **Why does `TvidReader` connect `StreamMetadata` to `audio.py`, `container.py`, `test_container.py`, `VideoLatentEngine`, `cli.py`, `container.py`, `transvid/__init__.py`, `test_video.py`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `AudioLatentEngine` (e.g. with `StreamMetadata` and `TvidReader`) actually correct?**
  _`AudioLatentEngine` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `VideoLatentEngine` be split into smaller, more focused modules?**
  _Cohesion score 0.14035087719298245 - nodes in this community are weakly interconnected._