"""Execute every notebook in the repository, in book order.

    python tools/run_all.py            # run everything, report pass/fail
    python tools/run_all.py --save     # also store the outputs in the notebooks (for GitHub previews)
    python tools/run_all.py 27 28      # only chapters whose file name starts with 27 or 28
    python tools/run_all.py solutions  # only the solution notebooks

Cells tagged "raises-exception" are the book's deliberate error examples: they are
expected to fail and do not stop the run.
"""
import os, sys, time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLOTLY_RENDERER", "png")   # static images that GitHub can display
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
save = "--save" in sys.argv

books = sorted((ROOT / "notebooks").glob("*/*.ipynb")) + sorted((ROOT / "solutions").glob("*.ipynb"))
if args:
    books = [p for p in books if any(p.name.startswith(a) or a in p.parts for a in args)]

failed = []
t_all = time.time()
for p in books:
    nb = nbformat.read(p, as_version=4)
    t0 = time.time()
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(p.parent)}})
    try:
        client.execute()
        status = "ok"
    except Exception as e:  # an unexpected error in a cell
        status = "FAILED"
        failed.append((p, str(e).strip().splitlines()[-1][:300] if str(e).strip() else repr(e)))
    print(f"{status:6} {time.time() - t0:6.1f}s  {p.relative_to(ROOT)}", flush=True)
    if save and status == "ok":
        for c in nb.cells:
            if c.cell_type == "code":
                c.metadata.pop("execution", None)
        nbformat.write(nb, p)

print(f"\n{len(books) - len(failed)}/{len(books)} notebooks ran cleanly in {time.time() - t_all:.0f}s")
for p, msg in failed:
    print(f"  {p.relative_to(ROOT)}: {msg}")
sys.exit(1 if failed else 0)
