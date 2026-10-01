"""Make the frozen model references in data/ loadable on this machine.

The frozen design artifacts record each model as ``<org>/<name>@<revision>``
(the Hugging Face repo id plus the exact commit). The stage scripts pass that
string straight to ``from_pretrained``, which treats an existing directory of
that name as a local checkpoint. This script downloads every recorded revision
into the Hugging Face cache and links ``<dest>/<org>/<name>@<revision>`` to the
cached snapshot, so the recorded string resolves without editing any artifact.

Run it once from the repository root; the default destinations cover scripts
run from the repository root and from ``code/`` (the sbatch wrappers ``cd``
there):

    python code/link_model_snapshots.py
"""
import argparse
import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
REF = re.compile(r'"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([0-9a-f]{40})')


def find_refs(data_dir: Path) -> set[tuple[str, str]]:
    refs = set()
    for path in data_dir.rglob("*.json"):
        refs.update(REF.findall(path.read_text(errors="ignore")))
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--dest", nargs="+", default=[str(ROOT), str(ROOT / "code")])
    parser.add_argument("--only", nargs="*", help="restrict to these repo ids")
    args = parser.parse_args()

    refs = sorted(find_refs(Path(args.data)))
    if args.only:
        refs = [r for r in refs if r[0] in args.only]
    for repo_id, revision in refs:
        local = snapshot_download(repo_id=repo_id, revision=revision)
        for dest in args.dest:
            link = Path(dest) / repo_id.split("/")[0] / f"{repo_id.split('/')[1]}@{revision}"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(local, link, target_is_directory=True)
        print(f"{repo_id}@{revision[:12]} -> {local}")


if __name__ == "__main__":
    main()
