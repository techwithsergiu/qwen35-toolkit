---
title: qwen35-upload
---


# qwen35-upload

Sync a local model directory with a HuggingFace Hub repo — both push and pull
directions. Does **not** load the model into memory — files are streamed
directly from disk. Works on machines with limited RAM/VRAM.

## Six sync modes

| Mode | Direction | When to use | What happens |
|---|---|---|---|
| `init` (default) | push | First push of a new model | Uploads full directory. Hub deduplicates by SHA256 — unchanged shards are not re-transferred |
| `files` | push | Patching specific files | Single atomic commit of the listed files only. Requires `--files` |
| `diff` | push | Incremental push after local edits | Compares local SHA256 vs remote; uploads new/changed files, deletes remote-only files |
| `check` | push dry-run | Inspect delta before pushing | Same per-file table as `diff`, nothing is uploaded or deleted |
| `pull` | pull | Sync local from Hub | Downloads new/changed files, removes local-only files. Large shards skipped if SHA256 matches |
| `fetch` | pull dry-run | Inspect delta before pulling | Same per-file table as `pull`, filesystem untouched |

## SHA256 and LFS behaviour

For `.safetensors`, `.gguf`, `.bin`, `.pt`, `.pth` shards the SHA256 is read
directly from the **LFS pointer** in the remote repo tree — no download needed
for the push side. This makes `check` and `diff` instant even for multi-shard
models.

Small non-LFS files (JSON configs, tokenizer files) are downloaded and hashed
on-the-fly (they're tiny so this is fast).

Local files are hashed in 8 MB chunks to avoid OOM on large shards.

## Default commit messages

| Mode | Default message |
|---|---|
| `init` | `Upload model` |
| `files` | `Update files` |
| `diff` | `Push changed files` |

Override with `--message`.

## Output format — push direction (`check` / `diff`)

```text
   FILE                                           SIZE  STATUS
   ─────────────────────────────────────────────────────────────────────
   config.json                                   2.1 KB  📝 changed
   model-00001-of-00002.safetensors              4.9 GB  ✅ unchanged  (a3f1c2d4e5b6…)
   model-00002-of-00002.safetensors              3.1 GB  ✅ unchanged  (9b8e7f6a5c4d…)
   tokenizer_config.json                         1.8 KB  ✅ unchanged  (c2d4e5b6a7f8…)
   README.md                                       512 B  ✨ new
   preprocessor_config.json                          —  🗑️  remote-only → delete

   ── Summary ─────────────────────────────────────────────────────────
      to upload  : 2
      to delete  : 1
      unchanged  : 3
```

## Output format — pull direction (`fetch` / `pull`)

```text
   FILE                                           SIZE  STATUS
   ─────────────────────────────────────────────────────────────────────
   config.json                                   2.1 KB  📝 changed
   model-00001-of-00002.safetensors              4.9 GB  ✅ unchanged  (a3f1c2d4e5b6…)
   model-00002-of-00002.safetensors              3.1 GB  ✅ unchanged  (9b8e7f6a5c4d…)
   tokenizer_config.json                         1.8 KB  ✅ unchanged  (c2d4e5b6a7f8…)
   some_local_extra.json                              —  🗑️  local-only → remove

   ── Summary ─────────────────────────────────────────────────────────
      to download : 1
      to remove   : 1
      unchanged   : 3
```

## CLI reference

```
qwen35-upload --local <dir> --repo <user/model> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--local` | `./Qwen3.5-0.8B-bnb-4bit` | Local model directory |
| `--repo` | `<your-hf-username>/Qwen3.5-0.8B-bnb-4bit` | HF Hub repo id |
| `--mode` | `init` | One of: `init` `files` `diff` `check` `pull` `fetch` |
| `--files` | — | Space-separated filenames for `--mode files` |
| `--message` | see table above | Custom commit message |
| `--private` | off | Create repo as private (only effective on first push) |
| `--hf-token` | env `HF_TOKEN` | HF access token; falls back to `HF_TOKEN` env var |

## Examples

```bash
# First push (BNB or f16 model)
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit

# First push — private repo
qwen35-upload \
    --local    ./Qwen3.5-0.8B-bnb-4bit \
    --repo     <your-hf-username>/Qwen3.5-0.8B-bnb-4bit \
    --private \
    --hf-token hf_...

# First push — GGUF quants (all quants in one repo)
qwen35-upload --local ./Qwen3.5-text-0.8B-GGUF --repo <your-hf-username>/Qwen3.5-text-0.8B-GGUF

# Patch specific files only (no re-upload of shards)
qwen35-upload \
    --local  ./Qwen3.5-0.8B-bnb-4bit \
    --repo   <your-hf-username>/Qwen3.5-0.8B-bnb-4bit \
    --mode   files \
    --files  README.md tokenizer_config.json

# Dry-run push — inspect delta without uploading
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode check

# Incremental push — new/changed files only, delete remote-only
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode diff

# Dry-run pull — inspect what would be downloaded/removed
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode fetch

# Pull new/changed files from Hub
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit --mode pull
```
