"""
upload.py
───────────────
Sync a local model directory with a Hugging Face Hub repo — both directions.

Does NOT load the model into memory — files are streamed directly from disk.
Works on machines with limited RAM/VRAM and handles large sharded models.

Six modes (controlled by --mode):

  PUSH direction (local → Hub):
  init   Upload the full directory. Creates the repo if needed.
         Re-uploading an already-present file is safe — the Hub deduplicates
         by SHA256, so unchanged shards are skipped automatically.

  files  Upload specific files only. Useful for targeted updates:
         e.g. after editing tokenizer_config.json or adding a README.

  diff   Compare SHA256 hashes and push only new / changed files.
         Also deletes remote files absent locally (mirrors local state).
         Prints a per-file log with size and hash status before committing.

  check  Dry-run of diff — print what would be pushed / deleted,
         but do NOT commit anything.

  PULL direction (Hub → local):
  pull   Download files that are new or changed on the Hub.
         Also deletes local files absent from the remote (mirrors remote state).
         Large files (safetensors, gguf) are skipped if SHA256 already matches.

  fetch  Dry-run of pull — print what would be downloaded / deleted locally,
         but do NOT touch the filesystem.

Typical use-cases:
  • First push of a BNB 4-bit model (output of convert_qwen35_to_bnb4bit.py)
  • First push of a merged f16 model  (output of merge_cpu.py)
  • Patch a tokenizer config or README without re-uploading model weights
  • Pull latest config / tokenizer fixes from Hub without re-downloading shards
  • Inspect push or pull delta before acting (check / fetch)

Usage:
  qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit --repo user/model
  qwen35-upload --local ./model --repo user/model --mode files --files tokenizer_config.json
  qwen35-upload --local ./model --repo user/model --mode diff
  qwen35-upload --local ./model --repo user/model --mode check
  qwen35-upload --local ./model --repo user/model --mode pull
  qwen35-upload --local ./model --repo user/model --mode fetch
  qwen35-upload --local ./model --repo user/model --private --hf-token hf_...
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi, create_repo,
    hf_hub_download,
)


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_LOCAL  = "./Qwen3.5-0.8B-bnb-4bit"
DEFAULT_REPO   = "techwithsergiu/Qwen3.5-0.8B-bnb-4bit"
DEFAULT_MODE   = "init"
DEFAULT_COMMIT = {
    "init":  "Upload model",
    "files": "Update files",
    "diff":  "Push changed files",
    "check": "",   # never used — check never commits
    "pull":  "",   # pull doesn't commit to Hub
    "fetch": "",   # fetch never touches anything
}

# File extensions treated as "large" model files — logged with extra detail.
LARGE_FILE_EXTS = {".safetensors", ".bin", ".gguf", ".ggml", ".pt", ".pth"}


# ── SHA256 helpers ─────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 of a local file, reading in chunks to avoid OOM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _truncate_path(path: str, max_len: int = 45) -> str:
    """Truncate a long path to max_len chars, prefixing with '…' when shortened."""
    if len(path) <= max_len:
        return path
    return "…" + path[-(max_len - 1):]


def fmt_size(n_bytes: int) -> str:
    """Human-readable file size: bytes → KB / MB / GB."""
    for unit, threshold in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n_bytes >= threshold:
            return f"{n_bytes / threshold:.1f} {unit}"
    return f"{n_bytes} B"


def remote_sha256_map(api: HfApi, repo_id: str) -> dict[str, str]:
    """
    Return {relative_path: sha256} for every file in the remote repo.

    LFS files (model shards) expose sha256 directly via the LFS pointer.
    Small non-LFS files (JSON configs, tokenizer files) are downloaded and
    hashed on-the-fly — they're tiny so this is fast.

    Returns an empty dict if the repo doesn't exist yet.
    """
    try:
        tree = api.list_repo_tree(repo_id, recursive=True)
    except Exception:
        return {}

    result: dict[str, str] = {}
    for entry in tree:
        # RepoFolder objects have no lfs attribute — skip them.
        if not hasattr(entry, "lfs"):
            continue

        if entry.lfs is not None:
            # LFS pointer carries the sha256 of the actual file content.
            result[entry.rfilename] = entry.lfs.sha256
        else:
            # Non-LFS file: download and hash.
            try:
                local = api.hf_hub_download(repo_id=repo_id, filename=entry.rfilename)
                result[entry.rfilename] = sha256_file(Path(local))
            except Exception:
                pass  # skip if unreachable

    return result


# ── Diff logic (shared by diff and check) ─────────────────────────────────────

def compute_diff(
    local_path: str,
    remote: dict[str, str],
) -> tuple[list[tuple[Path, str]], list[str], list[str]]:
    """
    Compare local files against the remote SHA256 map.

    Prints a per-file status line. Large model files (.safetensors, .gguf, …)
    get an extra line showing file size and the full hash comparison so it's
    easy to confirm exactly which shards changed.

    Args:
        local_path: Local model directory.
        remote:     {rel_path: sha256} from the remote repo (may be empty).

    Returns:
        (to_upload, to_delete, unchanged)
        to_upload : list of (local_path, repo_rel_path) for files that need pushing
        to_delete : list of repo_rel_path strings present remotely but absent locally
        unchanged : list of repo_rel_path strings for files already up to date
    """
    base        = Path(local_path)
    all_local   = sorted(f for f in base.rglob("*") if f.is_file())
    local_rels  = {f.relative_to(base).as_posix() for f in all_local}

    to_upload: list[tuple[Path, str]] = []
    to_delete: list[str]              = []
    unchanged: list[str]              = []

    print(f"   {'FILE':<45}  {'SIZE':>8}  STATUS")
    print(f"   {'─' * 45}  {'─' * 8}  {'─' * 30}")

    for local_file in all_local:
        rel        = local_file.relative_to(base).as_posix()
        size_bytes = local_file.stat().st_size
        size_str   = fmt_size(size_bytes)
        is_large   = local_file.suffix.lower() in LARGE_FILE_EXTS

        if rel not in remote:
            status = "✨ new"
            to_upload.append((local_file, rel))
        else:
            local_hash = sha256_file(local_file)
            if remote[rel] != local_hash:
                status = "📝 changed"
                to_upload.append((local_file, rel))
                if is_large:
                    status += (
                        f"\n   {'':45}  {'':8}"
                        f"  local  : {local_hash[:16]}…"
                        f"\n   {'':45}  {'':8}"
                        f"  remote : {remote[rel][:16]}…"
                    )
            else:
                status = "✅ unchanged"
                unchanged.append(rel)
                if is_large:
                    status += f"  ({local_hash[:16]}…)"

        name_col = _truncate_path(rel)
        print(f"   {name_col:<45}  {size_str:>8}  {status}")

    # Remote-only files — present on Hub but absent locally → mark for deletion.
    for rel in sorted(remote):
        if rel not in local_rels:
            name_col = _truncate_path(rel)
            print(f"   {name_col:<45}  {chr(8212):>8}  🗑️  remote-only → delete")
            to_delete.append(rel)

    return to_upload, to_delete, unchanged


# ── Upload modes ───────────────────────────────────────────────────────────────

def upload_init(
    api: HfApi,
    local_path: str,
    repo_id: str,
    commit_message: str,
) -> None:
    """
    Upload the entire local directory to the repo.

    The Hub deduplicates by SHA256, so unchanged files (e.g. large model
    shards) are not re-transferred even if the repo already exists.
    """
    print(f"⏫ Uploading full directory from '{local_path}' …")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )


def upload_files(
    api: HfApi,
    local_path: str,
    repo_id: str,
    filenames: list[str],
    commit_message: str,
) -> None:
    """
    Upload a specific list of files from local_path in a single atomic commit.

    Args:
        filenames: File names relative to local_path, e.g. ["tokenizer_config.json"].
    """
    base       = Path(local_path)
    operations = []
    missing    = []

    for name in filenames:
        src = base / name
        if not src.exists():
            missing.append(name)
            continue
        size_str = fmt_size(src.stat().st_size)
        print(f"   + {name}  ({size_str})")
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=src))

    if missing:
        print(f"   ⚠️  Not found locally (skipped): {missing}")
    if not operations:
        print("   ⚠️  Nothing to upload.")
        return

    print(f"\n⏫ Committing {len(operations)} file(s) …")
    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=commit_message,
    )


def upload_diff(
    api: HfApi,
    local_path: str,
    repo_id: str,
    commit_message: str,
    dry_run: bool = False,
) -> None:
    """
    Compare local files against the remote repo.

    Prints a per-file table with size and hash status, then either commits
    the changed files (dry_run=False) or exits without touching the repo
    (dry_run=True).

    Args:
        dry_run: If True, print the diff summary but do not upload anything.
    """
    print("🔍 Fetching remote file hashes …")
    remote = remote_sha256_map(api, repo_id)
    n_remote = len(remote)
    print(f"   {n_remote} file(s) found in remote repo\n")

    to_upload, to_delete, unchanged = compute_diff(local_path, remote)

    print(f"\n   ── Summary {'─' * 49}")
    print(f"      to upload  : {len(to_upload)}")
    print(f"      to delete  : {len(to_delete)}")
    print(f"      unchanged  : {len(unchanged)}")

    if dry_run:
        print("\n   ⚑  Dry-run — no files were uploaded or deleted.")
        return

    if not to_upload and not to_delete:
        print("\n   ✅ Remote is already up to date — nothing to push.")
        return

    operations = [
        CommitOperationAdd(path_in_repo=rel, path_or_fileobj=local_file)
        for local_file, rel in to_upload
    ] + [
        CommitOperationDelete(path_in_repo=rel)
        for rel in to_delete
    ]
    n_add = len(to_upload)
    n_del = len(to_delete)
    print(f"\n⏫ Committing {n_add} add(s) + {n_del} delete(s) …")
    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=commit_message,
    )


# ── Pull modes ─────────────────────────────────────────────────────────────────

def compute_pull_diff(
    local_path: str,
    remote: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Compare remote Hub files against the local directory (mirror of compute_diff).

    Prints a per-file status line for every remote file, then lists local-only
    files that would be deleted to mirror the remote state.

    Args:
        local_path: Local model directory.
        remote:     {rel_path: sha256} from the remote repo.

    Returns:
        (to_download, to_remove, unchanged)
        to_download : list of repo_rel_path strings to fetch from Hub
        to_remove   : list of local rel_path strings absent from remote
        unchanged   : list of rel_path strings already matching remote
    """
    base        = Path(local_path)
    local_rels  = {f.relative_to(base).as_posix() for f in base.rglob("*") if f.is_file()}

    to_download: list[str] = []
    to_remove:   list[str] = []
    unchanged:   list[str] = []

    print(f"   {'FILE':<45}  {'SIZE':>8}  STATUS")
    print(f"   {'─' * 45}  {'─' * 8}  {'─' * 30}")

    for rel, remote_hash in sorted(remote.items()):
        local_file = base / rel
        is_large   = Path(rel).suffix.lower() in LARGE_FILE_EXTS
        size_str   = "?"

        if not local_file.exists():
            status = "✨ new"
            to_download.append(rel)
        else:
            size_str   = fmt_size(local_file.stat().st_size)
            local_hash = sha256_file(local_file)
            if local_hash != remote_hash:
                status = "📝 changed"
                to_download.append(rel)
                if is_large:
                    status += (
                        f"\n   {'':45}  {'':8}"
                        f"  local  : {local_hash[:16]}…"
                        f"\n   {'':45}  {'':8}"
                        f"  remote : {remote_hash[:16]}…"
                    )
            else:
                status = "✅ unchanged"
                unchanged.append(rel)
                if is_large:
                    status += f"  ({local_hash[:16]}…)"

        name_col = _truncate_path(rel)
        print(f"   {name_col:<45}  {size_str:>8}  {status}")

    # Local-only files — absent from remote → mark for deletion.
    for rel in sorted(local_rels):
        if rel not in remote:
            name_col = _truncate_path(rel)
            print(f"   {name_col:<45}  {chr(8212):>8}  🗑️  local-only → remove")
            to_remove.append(rel)

    return to_download, to_remove, unchanged


def download_pull(
    api: HfApi,
    local_path: str,
    repo_id: str,
    dry_run: bool = False,
) -> None:
    """
    Download new / changed files from the Hub repo to local_path.

    Mirrors remote state: downloads new/changed files and removes local files
    that no longer exist on the Hub.  Large files whose SHA256 already matches
    are skipped — no re-download needed.

    Args:
        dry_run: If True, print the plan but do not touch the filesystem.
    """
    print("🔍 Fetching remote file hashes …")
    remote = remote_sha256_map(api, repo_id)
    if not remote:
        print("   ⚠️  Remote repo is empty or unreachable.")
        return
    print(f"   {len(remote)} file(s) found in remote repo\n")

    to_download, to_remove, unchanged = compute_pull_diff(local_path, remote)

    print(f"\n   ── Summary {'─' * 49}")
    print(f"      to download : {len(to_download)}")
    print(f"      to remove   : {len(to_remove)}")
    print(f"      unchanged   : {len(unchanged)}")

    if dry_run:
        print("\n   ⚑  Dry-run — filesystem was not modified.")
        return

    if not to_download and not to_remove:
        print("\n   ✅ Local directory already mirrors remote — nothing to do.")
        return

    base = Path(local_path)
    base.mkdir(parents=True, exist_ok=True)

    for rel in to_download:
        print(f"   ⬇️  {rel}")
        cached = hf_hub_download(repo_id=repo_id, filename=rel, token=api.token)
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dst)

    for rel in to_remove:
        target = base / rel
        if target.exists():
            target.unlink()
            print(f"   🗑️  Removed local: {rel}")


# ── Entry point ────────────────────────────────────────────────────────────────

def upload(
    local_path: str,
    repo_id: str,
    mode: str,
    files: list[str] | None,
    hf_token: str | None,
    private: bool,
    commit_message: str,
) -> None:
    """
    Main dispatcher — routes to the correct mode function.

    Args:
        local_path:     Local model directory.
        repo_id:        HF Hub repo id.
        mode:           "init" | "files" | "diff" | "check" | "pull" | "fetch"
        files:          File list for --mode files (ignored otherwise).
        hf_token:       HF access token; None uses cached credentials.
        private:        Create the repo as private if it doesn't exist.
        commit_message: Commit message shown on the Hub (push modes only).
    """
    api = HfApi(token=hf_token)

    # pull/fetch don't need to create a repo — they only read from it.
    if mode not in ("pull", "fetch"):
        create_repo(repo_id, token=hf_token, private=private, exist_ok=True)

    print(f"📦 Repo        : https://huggingface.co/{repo_id}")
    print(f"   mode       : {mode}")
    print(f"   local      : {local_path}\n")

    if mode == "init":
        upload_init(api, local_path, repo_id, commit_message)
    elif mode == "files":
        if not files:
            raise ValueError("--mode files requires --files <file1> [file2 …]")
        upload_files(api, local_path, repo_id, files, commit_message)
    elif mode == "diff":
        upload_diff(api, local_path, repo_id, commit_message, dry_run=False)
    elif mode == "check":
        upload_diff(api, local_path, repo_id, commit_message, dry_run=True)
    elif mode == "pull":
        download_pull(api, local_path, repo_id, dry_run=False)
    elif mode == "fetch":
        download_pull(api, local_path, repo_id, dry_run=True)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if mode not in ("check", "fetch"):
        print(f"\n✅ Done : https://huggingface.co/{repo_id}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Upload a local model directory to Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--local", default=DEFAULT_LOCAL,
        help="Path to the local model directory.",
    )
    ap.add_argument(
        "--repo", default=DEFAULT_REPO,
        help="Target HF Hub repo id, e.g. 'username/model-name'.",
    )
    ap.add_argument(
        "--mode", default=DEFAULT_MODE,
        choices=["init", "files", "diff", "check", "pull", "fetch"],
        help=(
            "Sync mode. "
            "PUSH: init=full upload; files=specific files; "
            "diff=push new/changed (+ delete remote-only); check=dry-run of diff. "
            "PULL: pull=download new/changed (+ delete local-only); fetch=dry-run of pull."
        ),
    )
    ap.add_argument(
        "--files", nargs="+", default=None, metavar="FILE",
        help="File names to upload relative to --local. Only used with --mode files.",
    )
    ap.add_argument(
        "--message", default=None,
        help="Commit message. Defaults per mode: init='Upload model', files='Update files', diff='Push changed files'.",
    )
    ap.add_argument(
        "--hf-token", default=None,
        help="HF access token. Omit to use credentials from `hf auth login`.",
    )
    ap.add_argument(
        "--private", action="store_true", default=False,
        help="Create the repo as private (no effect if it already exists).",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point for the qwen35-upload CLI command."""
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    mode = args.mode
    msg  = args.message or DEFAULT_COMMIT.get(mode, "")

    upload(
        local_path=args.local,
        repo_id=args.repo,
        mode=mode,
        files=args.files,
        hf_token=hf_token,
        private=args.private,
        commit_message=msg,
    )


if __name__ == "__main__":
    main()
