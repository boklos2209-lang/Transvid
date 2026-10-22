# Graph Report - Transvid  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 112 nodes · 228 edges · 8 communities (3 shown, 5 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 24 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `85c3c7ca`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- container.py
- StreamMetadata
- test_container.py
- VideoLatentEngine
- cli.py
- transvid

## God Nodes (most connected - your core abstractions)
1. `TvidReader` - 19 edges
2. `AudioLatentEngine` - 18 edges
3. `VideoLatentEngine` - 16 edges
4. `TvidWriter` - 15 edges
5. `StreamMetadata` - 12 edges
6. `cmd_compress()` - 9 edges
7. `cmd_decompress()` - 9 edges
8. `TestTvidContainer` - 6 edges
9. `build_decoder()` - 6 edges
10. `TestAudioLatentEngine` - 5 edges

## Surprising Connections (you probably didn't know these)
- `TestTvidContainer` --uses--> `StreamMetadata`  [INFERRED]
  tests/test_container.py → src/container.py
- `TestTvidContainer` --uses--> `TvidReader`  [INFERRED]
  tests/test_container.py → src/container.py
- `TestTvidContainer` --uses--> `TvidWriter`  [INFERRED]
  tests/test_container.py → src/container.py
- `TestAudioLatentEngine` --uses--> `AudioLatentEngine`  [INFERRED]
  tests/test_audio.py → src/audio.py
- `TestVideoLatentEngine` --uses--> `VideoLatentEngine`  [INFERRED]
  tests/test_video.py → src/video.py

## Import Cycles
- None detected.

## Communities (8 total, 5 thin omitted)

### Community 1 - "container.py"
Cohesion: 0.22
Nodes (5): Chunk, StreamMetadata, TvidReader, TvidWriter, TestTvidContainer

### Community 3 - "test_container.py"
Cohesion: 0.16
Nodes (5): build_decoder(), build_encoder(), Clamp, conv(), ResBlock

### Community 5 - "cli.py"
Cohesion: 0.38
Nodes (5): cmd_compress(), cmd_decompress(), cmd_info(), format_size(), main()

## Knowledge Gaps
- **1 isolated node(s):** `transvid`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 43 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AudioLatentEngine` connect `StreamMetadata` to `TvidWriter`, `container.py`, `test_container.py`, `VideoLatentEngine`, `cli.py`?**
  _High betweenness centrality (0.222) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `AudioLatentEngine` (e.g. with `VideoLatentEngine` and `TestAudioLatentEngine`) actually correct?**
  _`AudioLatentEngine` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `transvid` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `TvidWriter` be split into smaller, more focused modules?**
  _Cohesion score 0.12318840579710146 - nodes in this community are weakly interconnected._
- **Why does `VideoLatentEngine` connect `VideoLatentEngine` to `TvidWriter`, `container.py`, `StreamMetadata`, `test_container.py`, `cli.py`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `VideoLatentEngine` (e.g. with `AudioLatentEngine` and `TestVideoLatentEngine`) actually correct?**
  _`VideoLatentEngine` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `StreamMetadata` be split into smaller, more focused modules?**
  _Cohesion score 0.14619883040935672 - nodes in this community are weakly interconnected._