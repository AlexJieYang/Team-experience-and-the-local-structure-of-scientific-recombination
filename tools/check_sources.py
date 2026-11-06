"""Check source syntax and notebook hygiene without running research code."""
from pathlib import Path
import ast
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    counts = {"python": 0, "json": 0, "notebooks": 0}
    for name in sorted(set(names)):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
            counts["python"] += 1
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
            counts["json"] += 1
        elif path.suffix == ".ipynb":
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if document.get("nbformat") != 4 or not isinstance(document.get("cells"), list):
                raise ValueError(f"{name}: expected a version 4 notebook")
            metadata = document.get("metadata", {})
            language = metadata.get("kernelspec", {}).get("language", "") or metadata.get("language_info", {}).get("name", "")
            if any(key in metadata for key in ("widgets", "execution", "ExecuteTime")):
                raise ValueError(f"{name}: saved execution metadata")
            for index, cell in enumerate(document["cells"], start=1):
                if cell.get("cell_type") not in ("code", "markdown", "raw"):
                    raise ValueError(f"{name}: invalid cell {index}")
                source = cell.get("source", "")
                if isinstance(source, list) and all(isinstance(line, str) for line in source):
                    source = "".join(source)
                if not isinstance(source, str):
                    raise ValueError(f"{name}: invalid source in cell {index}")
                if any(key in cell.get("metadata", {}) for key in ("widgets", "execution", "ExecuteTime")):
                    raise ValueError(f"{name}: execution metadata in cell {index}")
                if cell["cell_type"] == "code":
                    if cell.get("outputs", []) or cell.get("execution_count") is not None:
                        raise ValueError(f"{name}: saved output or execution count in cell {index}")
                    if language.lower() == "python":
                        ast.parse(source, filename=f"{name}:cell {index}")
            counts["notebooks"] += 1
    print("Static source checks passed:", counts)


if __name__ == "__main__":
    main()
