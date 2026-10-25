#!/usr/bin/env python3
"""Download and verify a released DINO-WM checkpoint (PushT or Meta-World)."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
# env -> (url, sha256, size in bytes)
CHECKPOINTS = {
    "pusht": (
        "https://dl.fbaipublicfiles.com/jepa-wms/pt_dino-wm.pth.tar",
        "8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f",
        274_877_419,
    ),
    "metaworld": (
        "https://dl.fbaipublicfiles.com/jepa-wms/mw_dino-wm.pth.tar",
        "eab6e241f732db986f98ad224115ed7949ab4ff3f6cc8bc8f95b273cdb9e5d13",
        280_902_699,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=CHECKPOINTS, default="pusht")
    parser.add_argument("--output", type=Path, help="default: checkpoints/dino_wm_<env>.pth.tar")
    args = parser.parse_args()
    url, expected_sha256, expected_bytes = CHECKPOINTS[args.env]
    destination = (args.output or ROOT / "checkpoints" / f"dino_wm_{args.env}.pth.tar").resolve()

    if destination.is_file() and sha256(destination) == expected_sha256:
        print(f"Checkpoint already verified: {destination}")
        return
    if destination.exists():
        raise RuntimeError(
            f"Refusing to replace existing file with an unexpected hash: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "JEPA-Planning checkpoint downloader"},
    )
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)

        actual_size = partial.stat().st_size
        actual_hash = sha256(partial)
        if actual_size != expected_bytes or actual_hash != expected_sha256:
            raise RuntimeError(
                "Checkpoint verification failed: "
                f"size={actual_size}, sha256={actual_hash}"
            )
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print(f"Downloaded and verified checkpoint: {destination}")
    print(f"SHA-256: {expected_sha256}")


if __name__ == "__main__":
    main()
