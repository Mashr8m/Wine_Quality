"""Restore previous model artifacts locally; restart the service afterward."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FILES = ("model.joblib", "meta.json", "etalon.json")


def rollback(artifacts_dir: Path, version: str) -> dict:
    artifacts_dir = Path(artifacts_dir)
    source = artifacts_dir / "previous" / version
    for name in FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Нет резервного артефакта {source / name}")
    metadata = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    if metadata["model_version"] != version:
        raise ValueError("Несовпадение номера версии в meta.json")
    current_meta = artifacts_dir / "meta.json"
    if current_meta.is_file():
        current_version = json.loads(current_meta.read_text(encoding="utf-8"))["model_version"]
        if current_version != version:
            # Save a copy of the current version so the operation can be reversed.
            backup = artifacts_dir / "previous" / current_version
            backup.mkdir(parents=True, exist_ok=True)
            for name in FILES:
                if (artifacts_dir / name).exists():
                    shutil.copy2(artifacts_dir / name, backup / name)
    for name in FILES:
        shutil.copy2(source / name, artifacts_dir / name)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Вернуть предыдущую версию модели")
    parser.add_argument("--version", required=True, help="Пример: wine-logreg-1.0.0")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    args = parser.parse_args()
    meta = rollback(args.artifacts_dir, args.version)
    print(f"Откат выполнен: {meta['model_version']}. Перезапустите uvicorn для загрузки новой модели.")


if __name__ == "__main__":
    main()
