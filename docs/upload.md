---
title: qwen35-upload
---

# qwen35-upload

## Purpose

Synchronize a local model directory with a Hugging Face Hub repo (push and pull).
The command streams files from disk and does not load model weights into memory.

## When to use

- First upload of a new model repo.
- Incremental push after local file changes.
- Pull or preview remote changes before local sync.
- Patch specific files without re-uploading large shards.

## Syntax

```text
qwen35-upload --local <dir> --repo <user/model> [options]
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--local` | `./Qwen3.5-0.8B-bnb-4bit` | Local model directory |
| `--repo` | `<your-hf-username>/Qwen3.5-0.8B-bnb-4bit` | HF Hub repo id |
| `--mode` | `init` | `init` `files` `diff` `check` `pull` `fetch` |
| `--files` | — | Space-separated file list for `--mode files` |
| `--message` | mode default | Custom commit message |
| `--private` | off | Create repo as private (first push only) |
| `--hf-token` | env `HF_TOKEN` | HF token; falls back to env var |

## Modes

| Mode | Direction | When to use | Behavior |
|---|---|---|---|
| `init` (default) | push | First push | Upload full directory |
| `files` | push | Patch selected files | Commit listed files only (`--files` required) |
| `diff` | push | Incremental push | Upload changed/new, delete remote-only |
| `check` | push dry-run | Preview push delta | Same diff table, no changes applied |
| `pull` | pull | Sync local from Hub | Download changed/new, remove local-only |
| `fetch` | pull dry-run | Preview pull delta | Same pull table, no local changes |

## Mode decision flow

```text
1. First publish:
   - Use `init` (optionally `--private` on first repo creation).
2. Targeted patch:
   - Use `files` with explicit `--files`.
3. Incremental sync push:
   - Use `check` (dry-run), then `diff`.
4. Incremental sync pull:
   - Use `fetch` (dry-run), then `pull`.
```

Why it matters: dry-run before mutating sync prevents accidental upload/delete/remove operations.

## SHA256 and LFS behavior

- Large model artifacts (`.safetensors`, `.gguf`, `.bin`, `.pt`, `.pth`) use SHA256 from remote LFS pointers (no shard download for push-side comparison).
- Small files (JSON/tokenizer/docs) are downloaded and hashed when needed.
- Local hashing runs in chunks to avoid memory spikes on large files.

## Default commit messages

| Mode | Default message |
|---|---|
| `init` | `Upload model` |
| `files` | `Update files` |
| `diff` | `Push changed files` |

## Sync safety gates

```text
Gate 1 — Local readiness:
  - Local directory contains expected artifacts and no accidental temp files.

Gate 2 — Delta preview:
  - `check`/`fetch` output matches intended changes.

Gate 3 — Mutating sync:
  - Run `diff`/`pull` only after preview approval.

Gate 4 — Post-sync verify:
  - Re-run `check`/`fetch`; expected result is no further delta.
```

## Output formats

### Push side (check / diff)

```text
FILE                                               SIZE  STATUS
─────────────────────────────────────────────  ────────  ──────────────────────────────
.gitattributes                                   1.7 KB  ✅ unchanged
README.md                                        3.2 KB  ✅ unchanged
config.json                                      3.0 KB  ✅ unchanged
model.safetensors                              923.0 MB  ✅ unchanged  (95f507eb0fb2ba5a…)
tokenizer.json                                  19.1 MB  ✅ unchanged

── Summary ─────────────────────────────────────────────────
to upload  : 0
to delete  : 0
unchanged  : 15
```

Interpretation: push-side diff table is empty (repo already synchronized), so `check` and `diff` perform no upload/delete actions.

Stable fields: status categories and summary counters (`to upload`, `to delete`, `unchanged`).
Variable fields: hashes, file sizes, file count, exact file list.

### Pull side (fetch / pull)

```text
FILE                                               SIZE  STATUS
─────────────────────────────────────────────  ────────  ──────────────────────────────
.gitattributes                                   1.7 KB  ✅ unchanged
README.md                                        3.2 KB  ✅ unchanged
config.json                                      3.0 KB  ✅ unchanged
model.safetensors                              923.0 MB  ✅ unchanged  (95f507eb0fb2ba5a…)
tokenizer.json                                  19.1 MB  ✅ unchanged

── Summary ─────────────────────────────────────────────────
to download : 0
to remove   : 0
unchanged   : 15
```

Interpretation: pull-side table shows local and remote are already mirrored, so no downloads/removals are applied.

Stable fields: pull summary counters (`to download`, `to remove`, `unchanged`) and status semantics.
Variable fields: hashes, file sizes, file count, exact file list.

## Examples

```bash
# First push
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit
```

```bash
# First push as private
qwen35-upload \
  --local    ./Qwen3.5-0.8B-bnb-4bit \
  --repo     <your-hf-username>/Qwen3.5-0.8B-bnb-4bit \
  --private \
  --hf-token hf_...
```

```bash
# Patch specific files only
qwen35-upload \
  --local  ./Qwen3.5-0.8B-bnb-4bit \
  --repo   <your-hf-username>/Qwen3.5-0.8B-bnb-4bit \
  --mode   files \
  --files  README.md tokenizer_config.json
```

```bash
# Dry-run push delta
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode check
```

```bash
# Incremental push
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode diff
```

```bash
# Dry-run pull
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode fetch
```

```bash
# Pull changes from Hub
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode pull
```

## Edge cases / limitations

> [!WARNING]
> `--mode files` requires `--files`. Without explicit file list, no patch set can be computed.

- `--private` affects only first repo creation.
- Pull modes can delete local-only files (`pull`) by design.

## Related

- [Quickstart](quickstart.md)
- [Models](models.md)
- [Verify](verify.md)
