"""Put your GitHub username into every link (Colab badges, CI badge, citation).

    python tools/set_github_user.py your-username
"""
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit(__doc__)
user = sys.argv[1].strip().lstrip("@")
root = Path(__file__).resolve().parents[1]
n = 0
for p in root.rglob("*"):
    if p.is_file() and p.suffix in {".md", ".ipynb", ".py", ".cff", ".yml"} and ".git" not in p.parts:
        s = p.read_text(encoding="utf-8")
        if "YOUR-GITHUB-USERNAME" in s and p.name != "set_github_user.py":
            p.write_text(s.replace("YOUR-GITHUB-USERNAME", user), encoding="utf-8")
            n += 1
print(f"Updated {n} files with username '{user}'.")
