"""Build the GitHub repository for The Annotated ML Handbook: the complete book.

- notebooks/: every chapter as a full-text notebook (prose, definitions, tips, warnings,
  line-by-line explanation tables, runnable code)
- solutions/: worked solutions, full text
- scripts/: plain .py versions of the code
- book/: front and back matter as Markdown (preface, how to use, quick reference,
  glossary, index of definitions) and a table of contents
- exercises/, figures/, data/
"""
import json, os, re, shutil, sys, glob, html as _html
from pathlib import Path
import nbformat as nbf

sys.argv = [sys.argv[0]]
import build

ROOT = Path("repo/annotated-ml-handbook")
GH_USER = "Mahir-606"
REPO = "annotated-ml-handbook"
PAGES = f"https://{GH_USER.lower()}.github.io/{REPO}/"
BADGE = {"B": "Beginner", "I": "Intermediate", "A": "Advanced"}
ICON = {"def": "📖", "tip": "💡", "warn": "⚠️", "bug": "🐞", "research": "🔬", "note": "📝",
        "key": "🔑", "ex": "✍️", "sol": "✅"}
PART_DIR = {
    "1": "part-1-python", "2": "part-2-numpy", "3": "part-3-pandas", "4": "part-4-visualization",
    "5": "part-5-scikit-learn", "6": "part-6-tensorflow-keras", "7": "part-7-ml-for-research",
    "8": "part-8-debugging",
}
CALL = re.compile(r"^:::(\w+)[ \t]*([^\n]*)\n(.*?)\n:::[ \t]*$", re.M | re.S)
FENCE = re.compile(r"^```(\w+)([^\n]*)\n(.*?)\n```[ \t]*$", re.M | re.S)

SETUP = '''# Repository setup (not part of the book's examples): run this cell first.
import os, sys
from pathlib import Path

# Files the examples create (CSV, models, figures) go into ./outputs next to the notebook.
OUT = Path.cwd() if Path.cwd().name == "outputs" else Path.cwd() / "outputs"
OUT.mkdir(exist_ok=True)
os.chdir(OUT)

# insurance.csv is the practice dataset created in Chapter 11 and reused by later chapters.
# If it is not here yet, copy it from the repository's data/ folder (or download it in Colab).
if not Path("insurance.csv").exists():
    import shutil, urllib.request
    for base in OUT.parents:
        if (base / "data" / "insurance.csv").exists():
            shutil.copy(base / "data" / "insurance.csv", "insurance.csv")
            break
    else:
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/%s/"
                                       "annotated-ml-handbook/main/data/insurance.csv", "insurance.csv")
        except Exception as e:
            print("Could not fetch insurance.csv; run Chapter 11 first.", e)

# The figure style used in the book.
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams.update({
    "figure.figsize": (7.2, 4.0), "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=["#1f4fff", "#e0245e", "#0f9b62", "#ff6a00",
                                     "#7b3ff2", "#0091b3", "#d99a00", "#d42a2a"]),
    "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlelocation": "left",
    "font.size": 10.5, "legend.frameon": False, "image.cmap": "viridis",
})

# Plotly: interactive in Jupyter/Colab; tools/run_all.py stores static PNGs so GitHub can show them.
try:
    import plotly.io as pio
    pio.renderers.default = os.environ.get("PLOTLY_RENDERER") or (
        "colab" if "google.colab" in sys.modules else "notebook_connected")
except ImportError:
    pass''' % GH_USER


def slug(path):
    return re.sub(r"^\d+b?-", "", Path(path).stem)


# ---------------------------------------------------------------- collect chapters
files = sorted(glob.glob("chapters/*.md"))
CH, KEYMAP = [], {}
n, cur = 0, None
for f in files:
    meta, text = build.read_head(f)
    if meta["kind"] != "chapter":
        continue
    n += 1
    if meta["part"] in PART_DIR:
        cur = meta["part"]
    title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
    name = f"{n:02d}-{slug(f)}"
    rel = f"notebooks/{PART_DIR[cur]}/{name}.ipynb"
    CH.append(dict(file=f, num=n, title=title, part=cur, name=name, text=text, rel=rel))
    KEYMAP[build.file_key(f)] = (f"Chapter {n}", rel)

SOL = [("solutions-parts-1-4", "Solutions: Parts I–IV", "chapters/37-appendix-solutions-1.md"),
       ("solutions-parts-5-8", "Solutions: Parts V–VIII", "chapters/37b-appendix-solutions-2.md")]
PAGES_MD = [  # (file in book/, source, title)
    ("preface.md", "chapters/00g-preface.md", "Preface"),
    ("acknowledgements.md", "chapters/00h-acknowledgements.md", "Acknowledgements"),
    ("how-to-use-this-book.md", "chapters/00i-how-to-use.md", "How to use this book"),
    ("quick-reference.md", "chapters/35-appendix-quickref.md", "Appendix A: Quick-reference sheets"),
    ("glossary.md", "chapters/36-appendix-glossary.md", "Appendix B: Glossary and further reading"),
]
for fn, src, _ in PAGES_MD:
    KEYMAP[build.file_key(src)] = (_.split(":")[0], f"book/{fn}")
KEYMAP[build.file_key(SOL[0][2])] = ("Appendix C", f"solutions/{SOL[0][0]}.ipynb")
KEYMAP[build.file_key(SOL[1][2])] = ("Appendix D", f"solutions/{SOL[1][0]}.ipynb")


def sol_nb_for(num):
    return SOL[0][0] if num <= 18 else SOL[1][0]


def relink(target, from_file):
    return os.path.relpath(target, os.path.dirname(from_file) or ".")


def resolve(s, from_file):
    def sub(m):
        k = m.group(1)
        if k not in KEYMAP:
            return m.group(0)
        label, rel = KEYMAP[k]
        return f"[{label}]({relink(rel, from_file)})"
    return re.sub(r"\[\[ch:([\w-]+)\]\]", sub, s)


def badge(s):
    return re.sub(r"\s*\{([BIA])\}\s*$", lambda m: f" · *{BADGE[m.group(1)]}*", s.rstrip())


# ---------------------------------------------------------------- text conversion
DEFS = []   # (term, chapter label, file)


def callout_md(kind, title, body, from_file, label_ctx):
    lab = build.CALL_LABEL.get(kind, kind.title())
    head = f"**{ICON.get(kind, '📌')} {lab}**" + (f" · **{title}**" if title and title != lab else "")
    body = resolve(body.strip(), from_file)
    if kind == "def":
        for mm in re.finditer(r"^\*\*(.+?)\*\*", body, re.M):
            DEFS.append((mm.group(1).rstrip(":").strip(), label_ctx, from_file))
    if kind == "ex" and label_ctx.startswith("Chapter"):
        num = int(label_ctx.split()[1])
        sol = sol_nb_for(num)
        body += (f"\n\n*Worked solutions:* [{sol}.ipynb]({relink('solutions/' + sol + '.ipynb', from_file)}), "
                 f"section *{label_ctx}*.")
    lines = [head, ""] + body.split("\n")
    return "\n".join("> " + l if l.strip() else ">" for l in lines)


def clean_html(s):
    s = re.sub(r'<div class="route-pic"><!--ROUTE--></div>\s*', "", s)
    s = re.sub(r'<span class="badge \w">(.*?)</span>', r"*\1*", s)
    s = re.sub(r'<p class="sig">(.*?)<br><span>(.*?)</span></p>', r"*\1*, \2", s)
    return s


def explain_html(code):
    disp, notes = build.split_annotations(code)
    rows = []
    for line, note in zip(disp.split("\n"), notes):
        if not note:
            continue
        lead = len(line) - len(line.lstrip(" "))
        c = "&nbsp;" * lead + _html.escape(line.lstrip(" "))
        rows.append(f"<tr><td><code>{c}</code></td><td>{build.inline_md(note)}</td></tr>")
    if not rows:
        return ""
    return ("<details open><summary><b>What each line does</b></summary>\n\n"
            "<table>\n" + "\n".join(rows) + "\n</table>\n</details>")


def segments(text, from_file, label_ctx, sec_prefix=None):
    """Yield ('md', markdown) and ('code', lang, flags, kv, code) in reading order."""
    text = clean_html(text)
    text = re.sub(r"^# .+\n", "", text, count=1)              # title goes in the header
    text = CALL.sub(lambda m: callout_md(m.group(1), m.group(2).strip(), m.group(3),
                                         from_file, label_ctx), text)
    sec = [0, 0]
    pos = 0

    def prose(chunk):
        out = []
        for ln in chunk.split("\n"):
            m = re.match(r"^(#{2,3}) (.+)$", ln)
            if m:
                lvl, t = len(m.group(1)), m.group(2)
                if sec_prefix is not None:
                    if lvl == 2:
                        sec[0] += 1; sec[1] = 0
                        t = f"{sec_prefix}.{sec[0]} {t}"
                    else:
                        sec[1] += 1
                        t = f"{sec_prefix}.{sec[0]}.{sec[1]} {t}"
                ln = "#" * lvl + " " + badge(t)
            out.append(ln)
        return resolve("\n".join(out), from_file).strip()

    for m in FENCE.finditer(text):
        chunk = prose(text[pos:m.start()])
        if chunk:
            yield ("md", chunk)
        flags, kv = build.parse_info(m.group(2))
        yield ("code", m.group(1), flags, kv, m.group(3))
        pos = m.end()
    chunk = prose(text[pos:])
    if chunk:
        yield ("md", chunk)


def error_cell_py(src):
    if "'''" not in src and not src.endswith("\\"):
        lit = "r'''\n" + src + "\n'''"
    else:
        lit = repr(src)
    return ("# This example raises an error ON PURPOSE (the book explains the cause and the fix).\n"
            "# It runs inside exec() so the script prints the error and carries on.\n"
            "_example = " + lit + "\n"
            "try:\n    exec(_example)\nexcept Exception as _e:\n"
            "    print(f\"[expected error] {type(_e).__name__}: {_e}\")")


def colab(rel_nb):
    return f"https://colab.research.google.com/github/{GH_USER}/{REPO}/blob/main/{rel_nb}"


def build_notebook(title_md, segs, nav_md):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    cells = [nbf.v4.new_markdown_cell(title_md), nbf.v4.new_code_cell(SETUP)]
    py = ["# %%\n" + SETUP]
    md = []
    stats = dict(run=0, error=0, norun=0)

    def flush():
        if md:
            cells.append(nbf.v4.new_markdown_cell("\n\n".join(md)))
            md.clear()

    for s in segs:
        if s[0] == "md":
            md.append(s[1])
            continue
        _, lang, flags, kv, code = s
        ttl = kv.get("title")
        src = build.split_annotations(code)[0]
        exp = explain_html(code)
        if lang not in ("py", "bash"):
            md.append("```text\n" + code + "\n```")
            continue
        if ttl:
            md.append(f"**{ttl}**")
        if lang == "bash":
            md.append("Run in a terminal (not in Python):\n\n```bash\n" + src + "\n```")
            if exp:
                md.append(exp)
            continue
        if "norun" in flags:
            stats["norun"] += 1
            md.append("*Shown for reference, not run automatically (it waits for input, needs your own "
                      "files, or would change your system). Copy it into a cell to try it.*\n\n```python\n"
                      + src + "\n```")
            if exp:
                md.append(exp)
            continue
        if "error" in flags:
            stats["error"] += 1
            md.append("⚠️ *This cell raises an error **on purpose**, so you can see the real message.*")
        flush()
        cell = nbf.v4.new_code_cell(src)
        if "error" in flags:
            cell.metadata["tags"] = ["raises-exception"]
            py.append("# %%\n" + error_cell_py(src))
        else:
            stats["run"] += 1
            py.append("# %%\n" + src)
        cells.append(cell)
        if exp:
            md.append(exp)
    if nav_md:
        md.append(nav_md)
    flush()
    nb.cells = cells
    return nb, "\n\n\n".join(py) + "\n", stats


def build_markdown(title, segs):
    out = [f"# {title}\n"]
    for s in segs:
        if s[0] == "md":
            out.append(s[1] + "\n")
            continue
        _, lang, flags, kv, code = s
        if kv.get("title"):
            out.append(f"**{kv['title']}**\n")
        out.append(f"```{'python' if lang == 'py' else lang}\n{build.split_annotations(code)[0]}\n```\n")
        exp = explain_html(code)
        if exp:
            out.append(exp + "\n")
    return "\n".join(out)


# ---------------------------------------------------------------- generate
for d in ("notebooks", "scripts", "solutions", "exercises", "figures", "data", "book"):
    if (ROOT / d).exists():
        shutil.rmtree(ROOT / d)
for d in ("notebooks", "scripts", "solutions", "exercises", "data", "book"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)
shutil.copy("run_fresh/insurance.csv", ROOT / "data" / "insurance.csv")

ALL = dict(run=0, error=0, norun=0)
EX = []
for i, c in enumerate(CH):
    pd_ = PART_DIR[c["part"]]
    pr = build.PARTS[c["part"]]
    rel = c["rel"]
    prev_ = CH[i - 1] if i else None
    next_ = CH[i + 1] if i + 1 < len(CH) else None
    title_md = (f"<sub>**The Annotated ML Handbook** · Part {pr['roman']}: {pr['name']} · "
                f"by Mahir Absar Khan</sub>\n\n"
                f"# Chapter {c['num']}: {c['title']}\n\n"
                f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab(rel)}) "
                f"&nbsp; [📚 Contents]({relink('book/README.md', rel)})"
                f" &nbsp; [🌐 Web edition]({PAGES})\n\n"
                "*Run the setup cell first, then read and run the cells from top to bottom. "
                "Every code cell is followed by a table explaining what each line does.*")
    nav = ["---", ""]
    links = []
    if prev_:
        links.append(f"⬅️ [Chapter {prev_['num']}: {prev_['title']}]({relink(prev_['rel'], rel)})")
    links.append(f"[📚 Contents]({relink('book/README.md', rel)})")
    if next_:
        links.append(f"[Chapter {next_['num']}: {next_['title']}]({relink(next_['rel'], rel)}) ➡️")
    nav.append(" · ".join(links))
    segs = list(segments(c["text"], rel, f"Chapter {c['num']}", sec_prefix=c["num"]))
    exs = [m.group(3).strip() for m in CALL.finditer(c["text"]) if m.group(1) == "ex"]
    nb, py, st = build_notebook(title_md, segs, "\n".join(nav))
    for k in ALL:
        ALL[k] += st[k]
    c["stats"] = st
    (ROOT / "notebooks" / pd_).mkdir(exist_ok=True)
    nbf.write(nb, ROOT / rel)
    (ROOT / "scripts" / pd_).mkdir(exist_ok=True)
    head = (f'"""The Annotated ML Handbook, Chapter {c["num"]}: {c["title"]}\n\n'
            f'By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line\n'
            f'explanations is in {rel}.\n'
            f'Cells are marked "# %%" so you can run them one at a time in VS Code, PyCharm or Spyder.\n"""\n\n')
    (ROOT / "scripts" / pd_ / f"{c['name']}.py").write_text(head + py)
    EX.append((c, exs))

for name, title, f in SOL:
    meta, text = build.read_head(f)
    rel = f"solutions/{name}.ipynb"
    title_md = (f"<sub>**The Annotated ML Handbook** · by Mahir Absar Khan</sub>\n\n# {title}\n\n"
                f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab(rel)}) "
                f"&nbsp; [📚 Contents]({relink('book/README.md', rel)})")
    segs = list(segments(text, rel, "Solutions"))
    nb, py, st = build_notebook(title_md, segs, f"---\n\n[📚 Contents]({relink('book/README.md', rel)})")
    for k in ALL:
        ALL[k] += st[k]
    nbf.write(nb, ROOT / rel)
    (ROOT / "solutions" / f"{name}.py").write_text(
        f'"""The Annotated ML Handbook: {title}. By Mahir Absar Khan."""\n\n' + py)

# front and back matter as Markdown
for fn, src, title in PAGES_MD:
    meta, text = build.read_head(src)
    rel = f"book/{fn}"
    body = build_markdown(title, segments(text, rel, title.split(":")[0]))
    body += f"\n---\n\n[📚 Contents](README.md)\n"
    (ROOT / rel).write_text(body)

# index of definitions (from every Definitions box in the chapters)
seen = {}
for term, ctx, f in DEFS:
    if ctx.startswith("Chapter"):
        seen.setdefault(term.lower(), (term, ctx, f))
items = sorted(seen.values(), key=lambda t: re.sub(r"[^a-z0-9]", "", t[0].lower()))
idx = ["# Index of definitions\n",
       f"All {len(items)} terms defined in the *Definitions* boxes, alphabetically, with the chapter that "
       "defines them. The [glossary](glossary.md) gives one-line meanings.\n"]
letter = None
for term, ctx, f in items:
    L = re.sub(r"[^A-Za-z0-9]", "", term)[:1].upper() or "#"
    L = "0–9" if L.isdigit() else L
    if L != letter:
        letter = L
        idx.append(f"\n## {L}\n")
    idx.append(f"- **{term}** · [{ctx}]({relink(f, 'book/index-of-definitions.md')})")
idx.append("\n---\n\n[📚 Contents](README.md)\n")
(ROOT / "book" / "index-of-definitions.md").write_text("\n".join(idx))

# book/README.md: the table of contents
toc = ["# The Annotated ML Handbook\n",
       "**by Mahir Absar Khan** · First edition, 2026\n",
       "*For every learner who has stared at a traceback at 2 a.m., sighed, and tried one more time.*\n",
       f"Read it here on GitHub, chapter by chapter (the notebooks show the text, the code, the real "
       f"outputs and every figure), open any chapter in Colab to run it, or read the designed "
       f"[web edition]({PAGES}) and download the [PDF]({PAGES}The-Annotated-ML-Handbook.pdf).\n",
       "## Front matter\n",
       "- [Preface](preface.md)", "- [Acknowledgements](acknowledgements.md)",
       "- [How to use this book](how-to-use-this-book.md)\n"]
cur = None
for c in CH:
    if c["part"] != cur:
        cur = c["part"]
        pr = build.PARTS[cur]
        toc.append(f"\n## Part {pr['roman']}: {pr['name']}\n")
        if pr.get("blurb"):
            toc.append(f"*{pr['blurb']}*\n")
    secs = [re.sub(r"\s*\{[BIA]\}\s*$", "", h) for h in re.findall(r"^## (.+)$", c["text"], re.M)]
    toc.append(f"{c['num']}. **[{c['title']}](../{c['rel']})**  \n   <sub>{' · '.join(secs)}</sub>")
toc += ["\n## Appendices\n",
        "- **Appendix A** · [Quick-reference sheets](quick-reference.md)",
        "- **Appendix B** · [Glossary and further reading](glossary.md)",
        f"- **Appendix C** · [Solutions: Parts I–IV](../solutions/{SOL[0][0]}.ipynb)",
        f"- **Appendix D** · [Solutions: Parts V–VIII](../solutions/{SOL[1][0]}.ipynb)",
        "- [Index of definitions](index-of-definitions.md)",
        "- [All exercises on one page](../exercises/README.md)\n",
        "## About the author\n",
        "**Mahir Absar Khan** is an industrial and production engineer and a Research Assistant at the "
        "Computational Intelligence and Operations Laboratory. He researches machine learning and LLMs in "
        "supply chains, manufacturing, and healthcare, and has co-authored papers at ACM, ICML, IJCAI, and "
        "AAAI venues.\n"]
(ROOT / "book" / "README.md").write_text("\n".join(toc))

# exercises index
ex_md = ["# Try it yourself: all exercises\n",
         "Every chapter ends with a few exercises. They are collected here so you can practise in one place. "
         "Try each one before looking at the solutions.\n",
         "| Part | Solutions |", "|---|---|",
         "| I–IV (Chapters 1–18) | [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb) |",
         "| V–VIII (Chapters 19–34) | [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb) |\n"]
cur = None
for c, exs in EX:
    if c["part"] != cur:
        cur = c["part"]
        pr = build.PARTS[cur]
        ex_md.append(f"\n## Part {pr['roman']}: {pr['name']}\n")
    ex_md.append(f"### Chapter {c['num']}: {c['title']}\n")
    ex_md.append(f"Chapter: [{c['name']}.ipynb](../{c['rel']}) · Solutions: "
                 f"[{sol_nb_for(c['num'])}.ipynb](../solutions/{sol_nb_for(c['num'])}.ipynb)\n")
    for e in exs:
        ex_md.append(resolve(e, "exercises/README.md") + "\n")
(ROOT / "exercises" / "README.md").write_text("\n".join(ex_md))

# figures by chapter
html = Path("out/index.html").read_text()
arts = re.split(r'(?=<article class="(?:chapter|page)[^"]*" id="art-)', html)
nfig = 0
for a in arts:
    m = re.match(r'<article class="[^"]*" id="art-([\w-]+)"', a)
    if not m:
        continue
    aid = m.group(1)
    imgs = list(dict.fromkeys(re.findall(r'src="(img/f\d+\.png)"', a)))
    if not imgs:
        continue
    if aid.startswith("ch-"):
        d = ROOT / "figures" / CH[int(aid[3:]) - 1]["name"]
    elif aid in ("app-c", "app-d"):
        d = ROOT / "figures" / ("solutions-parts-1-4" if aid == "app-c" else "solutions-parts-5-8")
    else:
        d = ROOT / "figures" / aid
    d.mkdir(parents=True, exist_ok=True)
    for k, src in enumerate(imgs, 1):
        shutil.copy(Path("out") / src, d / f"fig-{k:02d}.png")
        nfig += 1

json.dump(dict(chapters=[dict(num=c["num"], title=c["title"], part=c["part"], name=c["name"], **c["stats"])
                         for c in CH], totals=ALL, figures=nfig, defs=len(items)),
          open("repo_manifest.json", "w"), indent=1)
print("chapters", len(CH), "totals", ALL, "figures", nfig, "defs", len(items))
