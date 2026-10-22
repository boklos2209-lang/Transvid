# Graph Report - Transvid  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 29 nodes · 51 edges · 4 communities (1 shown, 3 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- TvidWriter
- StreamMetadata

## God Nodes (most connected - your core abstractions)
1. `TvidWriter` - 11 edges
2. `StreamMetadata` - 8 edges
3. `TvidReader` - 8 edges
4. `TestTvidContainer` - 6 edges
5. `Chunk` - 3 edges
6. `Writes a .tvid binary container.` - 1 edges
7. `Transvid Container Specification (.tvid)…` - 1 edges
8. `Reads and parses a .tvid binary container.` - 1 edges
9. `Unit tests for the .tvid binary container format. Tests serialization, chunk…` - 1 edges

## Surprising Connections (you probably didn't know these)
- `TestTvidContainer` --uses--> `TvidWriter`  [INFERRED]
  tests/test_container.py → src/container.py
- `TestTvidContainer` --uses--> `StreamMetadata`  [INFERRED]
  tests/test_container.py → src/container.py
- `TestTvidContainer` --uses--> `TvidReader`  [INFERRED]
  tests/test_container.py → src/container.py

## Import Cycles
- None detected.

## Communities (4 total, 3 thin omitted)

### Community 2 - "StreamMetadata"
Cohesion: 0.43
Nodes (3): StreamMetadata, TvidReader, TestTvidContainer

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TvidWriter` connect `TvidWriter` to `container.py`, `StreamMetadata`, `test_container.py`?**
  _High betweenness centrality (0.334) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `TestTvidContainer` (e.g. with `StreamMetadata` and `TvidReader`) actually correct?**
  _`TestTvidContainer` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Why does `TvidReader` connect `StreamMetadata` to `TvidWriter`, `container.py`, `test_container.py`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `StreamMetadata` connect `StreamMetadata` to `TvidWriter`, `container.py`, `test_container.py`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._