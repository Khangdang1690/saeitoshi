"""``hf://`` URL resolver.

Lazy-imports ``huggingface_hub`` so the dependency is optional. Install with
``pip install saeitoshi[hf]``.
"""

from __future__ import annotations

from pathlib import Path

_ALLOW_PATTERNS = [
    "**/cfg.json",
    "**/sae_weights.safetensors",
    "**/sae.safetensors",
    "**/weights.safetensors",
]


def parse_url(url: str) -> tuple[str, str | None, str | None]:
    """Parse ``hf://repo_id[@revision][/subfolder]``.

    Returns ``(repo_id, revision, subfolder)``. ``revision`` and
    ``subfolder`` may be ``None``.
    """
    if not url.startswith("hf://"):
        raise ValueError(f"expected hf:// URL, got {url!r}")
    body = url[len("hf://"):]
    revision: str | None = None
    if "@" in body:
        repo_part, _, rest = body.partition("@")
        rev_and_subfolder = rest.split("/", 1)
        revision = rev_and_subfolder[0]
        body = repo_part + ("/" + rev_and_subfolder[1] if len(rev_and_subfolder) > 1 else "")
    # body is now `repo_id` or `repo_id/subfolder...`
    parts = body.split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            f"hf:// URL must include `org/repo` (got {url!r})"
        )
    repo_id = "/".join(parts[:2])
    subfolder = parts[2] if len(parts) >= 3 and parts[2] else None
    return repo_id, revision, subfolder


def resolve(url: str) -> Path:
    """Download (or read cached) the SAE from HuggingFace Hub. Returns a
    local path to the SAE root directory.

    Optional dep: ``huggingface_hub``. Install with ``pip install
    saeitoshi[hf]``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "hf:// URLs require huggingface_hub. Install with "
            "`pip install \"saeitoshi[hf]\"`."
        ) from e

    repo_id, revision, subfolder = parse_url(url)
    local = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=_ALLOW_PATTERNS,
        )
    )
    if subfolder:
        local = local / subfolder
    return local
