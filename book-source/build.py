#!/usr/bin/env python3
"""Build the Annotated ML Handbook.

chapters/*.md  ->  out/index.html (single self-contained page)

Every ```py block is EXECUTED in order (one namespace per chapter file);
stdout, warnings, errors and matplotlib figures are captured and embedded,
so every output printed in the book is real.
"""
import os, re, io, sys, json, html, glob, time, base64, shlex, warnings, traceback, contextlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "0")

import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from pygments.lexers import PythonLexer, BashLexer
from pygments.token import Token

ROOT = os.path.dirname(os.path.abspath(__file__))
CH_DIR = os.path.join(ROOT, "chapters")
RUN_DIR = os.environ.get("BOOK_RUN_DIR", os.path.join(ROOT, "run"))
OUT = os.path.join(ROOT, "out")
ONLY = sys.argv[1:]  # optional: build only files whose name contains one of these

# ------------------------------------------------------------------ parts
PARTS = {
    "0": dict(roman="", name="Start here", short="Start", color="#1f4fff", on="#fff", blurb=""),
    "1": dict(roman="I", name="Python", short="Python", color="#1f4fff", on="#fff",
              blurb="The language itself: values, data structures, control flow, functions, classes, errors, files, and the Pythonic toolkit every ML codebase is built from."),
    "2": dict(roman="II", name="NumPy", short="NumPy", color="#7b3ff2", on="#fff",
              blurb="Fast n-dimensional arrays: shapes, indexing, vectorization, broadcasting, linear algebra and random numbers. This is the maths engine under pandas, scikit-learn and TensorFlow."),
    "3": dict(roman="III", name="Pandas", short="Pandas", color="#e0245e", on="#fff",
              blurb="Tables you can reason about: loading, inspecting, selecting, cleaning, reshaping, joining and summarising real-world data before it ever meets a model."),
    "4": dict(roman="IV", name="Visualization", short="Visualize", color="#0091b3", on="#fff",
              blurb="Seeing your data and your models: Matplotlib's figure anatomy, Seaborn's statistical plots, choosing the right chart, and exporting publication-quality figures."),
    "5": dict(roman="V", name="scikit-learn", short="scikit-learn", color="#0f9b62", on="#fff",
              blurb="Classical machine learning done properly: the estimator API, preprocessing pipelines, regression, classification, model selection, clustering, dimensionality reduction and interpretation."),
    "6": dict(roman="VI", name="TensorFlow & Keras", short="TensorFlow", color="#ff6a00", on="#fff",
              blurb="Deep learning from tensors and automatic differentiation up to dense, convolutional and recurrent networks, callbacks, custom training loops and saving models."),
    "7": dict(roman="VII", name="ML for Research", short="Research", color="#d99a00", on="#16130a",
              blurb="A step-by-step protocol for using ML in a research project: from question and data to validated, interpreted, reproducible results and a written Methods section."),
    "8": dict(roman="VIII", name="Debugging", short="Debugging", color="#d42a2a", on="#fff",
              blurb="When you are stuck: how to read tracebacks, an encyclopedia of real error messages with causes and fixes, and a symptom-to-cause guide for models that misbehave."),
    "9": dict(roman="", name="Appendices", short="Appendix", color="#4a5568", on="#fff",
              blurb="Dense quick-reference sheets, a model-and-metric chooser, and a glossary."),
}
ROUTE = ["1", "2", "3", "4", "5", "6", "7", "8"]

BOOK_TITLE = "The Annotated ML Handbook"
BOOK_SUB = "Python · NumPy · Pandas · Matplotlib · scikit-learn · TensorFlow"
BOOK_TAG = "From your first line of code to machine learning you can publish. Every line explained, every output real."
AUTHOR = "Mahir Absar Khan"

# ------------------------------------------------------------------ helpers
def esc(s):
    return html.escape(s, quote=True)

def inline_md(s):
    """Tiny inline markdown for explanation cells: `code`, **bold**, *italic*."""
    codes = []
    def keep(m):
        codes.append("<code>" + esc(m.group(1)) + "</code>")
        return f"\x00{len(codes)-1}\x00"
    s = re.sub(r"`([^`]+)`", keep, s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    s = s.replace("→", "&rarr;")
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)

TOKCLS = {
    Token.Keyword: "k", Token.Keyword.Constant: "kc", Token.Operator.Word: "k",
    Token.Name.Builtin: "b", Token.Name.Builtin.Pseudo: "b", Token.Name.Exception: "b",
    Token.Name.Function: "f", Token.Name.Function.Magic: "f", Token.Name.Class: "cl",
    Token.Name.Decorator: "d", Token.Literal.String: "s", Token.Literal.Number: "n",
    Token.Comment: "c", Token.Name.Namespace: "ns", Token.Name.Variable: "v",
}

def tok_class(tt):
    while tt is not None and tt is not Token:
        if tt in TOKCLS:
            return TOKCLS[tt]
        tt = tt.parent
    return None

def highlight_lines(code, lang):
    lexer = PythonLexer() if lang == "py" else BashLexer()
    toks = list(lexer.get_tokens(code))
    lines = [[]]
    for i, (tt, val) in enumerate(toks):
        c = tok_class(tt)
        if c is None and tt in Token.Name and lang == "py":
            nxt = toks[i + 1][1] if i + 1 < len(toks) else ""
            if nxt.startswith("("):
                c = "f"
        parts = val.split("\n")
        for j, p in enumerate(parts):
            if j > 0:
                lines.append([])
            if p:
                lines[-1].append(f'<span class="{c}">{esc(p)}</span>' if c else esc(p))
    n = code.count("\n") + 1
    return ["".join(l) for l in lines[:n]]

ANN = re.compile(r"(^|\s)#>\s?(.*)$")

def split_annotations(code):
    """Return (display_code, [explanation per display line])."""
    disp, notes = [], []
    for raw in code.split("\n"):
        m = ANN.search(raw)
        if m:
            src = raw[: m.start()].rstrip()
            note = m.group(2).strip()
            if src.strip() == "":           # pure explanation line -> attach to previous
                if notes:
                    notes[-1] = (notes[-1] + " " + note).strip()
                continue
            disp.append(src); notes.append(note)
        else:
            disp.append(raw.rstrip()); notes.append("")
    while disp and disp[-1].strip() == "":
        disp.pop(); notes.pop()
    return "\n".join(disp), notes

# ------------------------------------------------------------------ execution
BOOK_COLORS = ["#1f4fff", "#e0245e", "#0f9b62", "#ff6a00", "#7b3ff2", "#0091b3", "#d99a00", "#d42a2a"]

def book_style():
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.figsize": (7.2, 4.0), "figure.dpi": 100,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=BOOK_COLORS),
        "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlelocation": "left",
        "font.size": 10.5, "legend.frameon": False, "image.cmap": "viridis",
    })

class Runner:
    def __init__(self):
        self.ns = {"__name__": "__main__"}
        self.figs = []
        book_style()

    def _grab(self, *a, **k):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=130, bbox_inches="tight", facecolor="white")
            self.figs.append(b.getvalue())
        plt.close("all")

    def run(self, code, label):
        self.figs = []
        self.plys = []
        try:
            import plotly.basedatatypes as _pb
            def _pshow(fig, *a, **k):
                w = fig.layout.width or 900
                h = fig.layout.height or 480
                png = fig.to_image(format="png", width=w, height=h, scale=2)
                self.plys.append((fig.to_json(), png, w, h))
            _pb.BaseFigure.show = _pshow
        except ImportError:
            pass
        def _show(*a, **k):
            self._grab()
        plt.show = _show
        out = io.StringIO()
        err = None
        cwd = os.getcwd()
        os.chdir(RUN_DIR)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    exec(compile(code, "<cell>", "exec"), self.ns)
                except BaseException as e:  # noqa
                    err = e
        os.chdir(cwd)
        self._grab()
        wmsgs = []
        for w in wl:
            wmsgs.append(f"{w.category.__name__}: {w.message}")
        return out.getvalue(), err, wmsgs, list(self.figs), list(self.plys)

def format_error(err, code):
    tb = traceback.extract_tb(err.__traceback__)
    lines = ["Traceback (most recent call last):"]
    src = code.split("\n")
    lib_frames = 0
    for fr in tb:
        if fr.filename == "<cell>":
            if lib_frames:
                lines.append("  ...")
                lib_frames = 0
            lines.append(f'  File "<cell>", line {fr.lineno}, in {fr.name}')
            if 0 < fr.lineno <= len(src):
                lines.append("    " + ANN.sub("", src[fr.lineno - 1]).strip())
        elif fr.filename != "<string>" and "build.py" not in fr.filename:
            lib_frames += 1
    if lib_frames:
        lines.append(f"  ... ({lib_frames} frames inside the library)")
    msg = re.sub(r"\x1b\[[0-9;]*m", "", "".join(traceback.format_exception_only(type(err), err)).rstrip())   # strip ANSI colours
    msg_lines = msg.split("\n")
    if len(msg_lines) > 14:
        msg_lines = msg_lines[:12] + ["    ..."]
    return "\n".join(lines + msg_lines)

def trim(text, maxlines=70):
    ls = text.rstrip("\n").split("\n")
    if len(ls) > maxlines:
        ls = ls[: maxlines - 12] + [f"... ({len(ls) - maxlines + 12} more lines) ..."] + ls[-11:]
    return "\n".join(ls)

# ------------------------------------------------------------------ rendering blocks
LANG_NAME = {"py": "Python", "bash": "Terminal", "text": "Text"}

def render_annotated(code, lang, title=""):
    disp, notes = split_annotations(code)
    hl = highlight_lines(disp, lang)
    has_notes = any(notes)
    rows = []
    for i, (h, n) in enumerate(zip(hl, notes), 1):
        rows.append(
            f'<div class="r{" e" if not n else ""}"><span class="ln">{i}</span>'
            f'<code class="src">{h or " "}</code>'
            + (f'<span class="ex">{inline_md(n)}</span>' if has_notes else "")
            + "</div>"
        )
    cap = f'<span class="ct">{inline_md(title)}</span>' if title else ""
    return (
        f'<figure class="code{" plain" if not has_notes else ""}">'
        f'<figcaption><span class="lang">{LANG_NAME.get(lang, lang)}</span>{cap}'
        f'{"<span class=exh>What each line does</span>" if has_notes else ""}'
        f'<button class="copy" type="button">Copy</button></figcaption>'
        f'<div class="ann">{"".join(rows)}</div>'
        f"<textarea hidden>{esc(disp)}</textarea></figure>"
    )

def render_output(text, kind="out"):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    label = {"out": "Output", "err": "Error (this is expected)", "warn": "Warning"}[kind]
    return f'<div class="out {kind}"><span class="ol">{label}</span><pre>{esc(text)}</pre></div>'

IMG = [0]
def save_img(data):
    IMG[0] += 1
    rel = f"img/f{IMG[0]:03d}.png"
    with open(os.path.join(OUT, rel), "wb") as fh:
        fh.write(data)
    return rel

def render_figs(figs, caption):
    h = []
    for i, f in enumerate(figs):
        cap = f"<figcaption>{inline_md(caption)}</figcaption>" if caption and i == len(figs) - 1 else ""
        h.append(f'<figure class="plot"><img loading="lazy" alt="{esc(caption or "Figure produced by the code above")}" src="{save_img(f)}">{cap}</figure>')
    return "".join(h)

PLY = [0]
def render_plotly(plys, caption):
    h = []
    for i, (js, png, w, hh) in enumerate(plys):
        PLY[0] += 1
        fid = f"ply{PLY[0]}"
        cap = f"<figcaption>{inline_md(caption)}</figcaption>" if caption and i == len(plys) - 1 else ""
        h.append(f'<figure class="plot ply"><div class="plive" id="{fid}" style="aspect-ratio:{w}/{hh}"></div>'
                 f'<script type="application/json" data-for="{fid}">{js.replace("</", "<\\/")}</script>'
                 f'<img class="pimg" alt="{esc(caption or "Plotly chart")}" src="{save_img(png)}">'
                 f'<span class="plyhint">Interactive: hover, zoom, click the legend</span>{cap}</figure>')
    return "".join(h)

FENCE = re.compile(r"^```(\w+)([^\n]*)\n(.*?)\n```[ \t]*$", re.M | re.S)
CALL = re.compile(r"^:::(\w+)[ \t]*([^\n]*)\n(.*?)\n:::[ \t]*$", re.M | re.S)
DEFS, DEF_IDS, REFS, CUR = [], set(), {}, {"label": "", "color": "#1f4fff", "id": ""}
CALL_LABEL = {"def": "Definitions", "sol": "Solution", "tip": "Tip", "warn": "Pitfall", "bug": "Bug hunt", "note": "Note",
              "research": "Research practice", "key": "Key idea", "ex": "Try it yourself"}

STATS = {"blocks": 0, "run": 0, "figs": 0, "errors_shown": 0}

def parse_info(info):
    flags, kv = set(), {}
    for t in shlex.split(info):
        if "=" in t:
            k, v = t.split("=", 1); kv[k] = v
        else:
            flags.add(t)
    return flags, kv

def process_code(text, runner, fname, log):
    blocks = []
    def sub(m):
        lang, info, code = m.group(1), m.group(2), m.group(3)
        flags, kv = parse_info(info)
        html_parts = []
        STATS["blocks"] += 1
        if lang == "output":
            html_parts.append(render_output(code))
        elif lang == "text":
            html_parts.append(f'<pre class="txt">{esc(code)}</pre>')
        elif lang in ("py", "bash"):
            show_code = "hide" not in flags and "figonly" not in flags
            if show_code:
                html_parts.append(render_annotated(code, lang, kv.get("title", "")))
            if lang == "py" and "norun" not in flags:
                STATS["run"] += 1
                t0 = time.time()
                out, err, wmsgs, figs, plys = runner.run(code, fname)
                dt = time.time() - t0
                if dt > 20:
                    log.append(f"  slow block ({dt:.0f}s) in {fname}: {code[:60]!r}")
                if err is not None and "error" not in flags:
                    print(f"\n!!! UNEXPECTED ERROR in {fname}\n{code}\n{format_error(err, code)}\n{out}")
                    sys.exit(1)
                if err is None and "error" in flags:
                    print(f"\n!!! block marked error did not raise in {fname}\n{code}")
                    sys.exit(1)
                if wmsgs and "warn" not in flags:
                    log.append(f"  warnings in {fname}: {wmsgs[:2]} :: {code[:50]!r}")
                if out.strip() and "noout" not in flags:
                    html_parts.append(render_output(trim(out)))
                if "warn" in flags and wmsgs:
                    seen = []
                    for w in wmsgs:
                        if w not in seen: seen.append(w)
                    html_parts.append(render_output(trim("\n".join(seen), 20), "warn"))
                if err is not None:
                    STATS["errors_shown"] += 1
                    html_parts.append(render_output(format_error(err, code), "err"))
                if figs:
                    STATS["figs"] += len(figs)
                    html_parts.append(render_figs(figs, kv.get("caption", "")))
                if plys:
                    STATS["figs"] += len(plys)
                    html_parts.append(render_plotly(plys, kv.get("caption", "")))
        else:
            html_parts.append(f'<pre class="txt">{esc(code)}</pre>')
        blocks.append("".join(html_parts))
        return f'\n\n<div data-ph="{len(blocks)-1}"></div>\n\n'
    text = FENCE.sub(sub, text)
    return text, blocks

MD_EXT = ["tables", "sane_lists"]

def md(s):
    return markdown.markdown(s, extensions=MD_EXT, output_format="html")

def process_callouts(text, blocks):
    def sub(m):
        kind, title, body = m.group(1), m.group(2).strip(), m.group(3)
        lab = CALL_LABEL.get(kind, kind.title())
        t = f"<b class=cl-t>{inline_md(title)}</b>" if title else ""
        inner = md(body)
        if kind == "def":
            def addid(mm):
                term = strip_tags(mm.group(1)).rstrip(":").strip()
                slug = "def-" + re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
                while slug in DEF_IDS:
                    slug += "-x"
                DEF_IDS.add(slug)
                DEFS.append((term, slug, CUR["label"], CUR["color"], CUR["id"]))
                return f'<p id="{slug}" class="dterm"><strong>{mm.group(1)}</strong>'
            inner = re.sub(r"<p><strong>(.*?)</strong>", addid, inner)
        blocks.append(f'<aside class="call {kind}"><span class="cl-l">{lab}</span>{t}{inner}</aside>')
        return f'\n\n<div data-ph="{len(blocks)-1}"></div>\n\n'
    return CALL.sub(sub, text)

def fill_placeholders(h, blocks):
    pat = re.compile(r'(?:<p>)?<div data-ph="(\d+)"></div>(?:</p>)?')
    for _ in range(4):
        h2 = pat.sub(lambda m: blocks[int(m.group(1))], h)
        if h2 == h:
            break
        h = h2
    return h

BADGE = {"B": ("b", "Beginner"), "I": ("i", "Intermediate"), "A": ("a", "Advanced")}

def badge_sub(s):
    return re.sub(r"\s*\{([BIA])\}", lambda m: f' <span class="badge {BADGE[m.group(1)][0]}">{BADGE[m.group(1)][1]}</span>', s)

def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()

# ------------------------------------------------------------------ chapter
def file_key(path):
    return re.sub(r"^\d+[a-z]?-", "", os.path.basename(path)[:-3])

def read_head(path):
    text = open(path, encoding="utf-8").read()
    meta = {"kind": "chapter", "part": None, "class": "", "toc": ""}
    m = re.match(r"((?:@\w+[^\n]*\n)+)", text)
    if m:
        for line in m.group(1).strip().split("\n"):
            k, _, v = line[1:].partition(" ")
            meta[k] = v.strip()
        text = text[m.end():]
    return meta, text

def prepass(files):
    ch = app = 0
    for f in files:
        meta, text = read_head(f)
        key = file_key(f)
        if meta["kind"] == "chapter":
            ch += 1; REFS[key] = (f"Chapter {ch}", f"ch-{ch}")
        elif meta["kind"] == "appendix":
            app += 1; L = "ABCDEFGH"[app - 1]; REFS[key] = (f"Appendix {L}", f"app-{L.lower()}")

def resolve_refs(text, fname):
    def sub(m):
        k = m.group(1)
        if k not in REFS:
            print(f"!!! unknown cross-reference [[ch:{k}]] in {fname}"); sys.exit(1)
        lab, cid = REFS[k]
        return f"[{lab}](#{cid})"
    return re.sub(r"\[\[ch:([\w-]+)\]\]", sub, text)

def render_chapter(path, runner_factory, state, log):
    meta, text = read_head(path)
    kind, part = meta["kind"], meta["part"]
    fname = os.path.basename(path)
    if part:
        CUR["color"] = PARTS[part]["color"]
    if kind == "chapter":
        state["ch"] += 1; num = str(state["ch"]); kicker = f"Chapter {num}"; cid = f"ch-{num}"
    elif kind == "appendix":
        state["app"] += 1; num = "ABCDEFGH"[state["app"] - 1]; kicker = f"Appendix {num}"; cid = f"app-{num.lower()}"
    else:
        state["fr"] += 1; num = ""; kicker = ""; cid = "pg-" + file_key(path)
    CUR["label"], CUR["id"] = (kicker or "Front matter"), cid
    toc = {"id": cid, "kicker": kicker, "title": "", "secs": [], "kind": kind, "toc": meta.get("toc", "")}
    if kind == "contents":
        return part, toc, "<!--CONTENTS-->"
    if kind == "defindex":
        toc["title"] = "Index of definitions"
        toc["id"] = "pg-defindex"
        return part, toc, "<!--DEFINDEX-->"
    text = resolve_refs(text, fname)
    if kind == "page":
        h = md(text)
        t = re.search(r"<h1>(.*?)</h1>", h, re.S)
        toc["title"] = strip_tags(t.group(1)) if t else file_key(path)
        h = re.sub(r"<h1>(.*?)</h1>", lambda m: f'<h1 id="{cid}">{m.group(1)}</h1>', h, count=1)
        if not t:
            h = f'<span id="{cid}"></span>' + h
        return part, toc, f'<article class="page {meta.get("class", "")}" id="art-{cid}">{h}</article>'
    runner = runner_factory()
    text, blocks = process_code(text, runner, fname, log)
    text = process_callouts(text, blocks)
    h = md(text)
    h = fill_placeholders(h, blocks)
    h = re.sub(r"<li>\[ \]\s*", '<li class="chk"><span class="box"></span>', h)
    h = re.sub(r"<table>", '<div class="tablewrap"><table>', h)
    h = h.replace("</table>", "</table></div>")
    sec = [0]
    sub = [0]

    def h1(m):
        t = m.group(1); toc["title"] = strip_tags(t)
        kk = f'<p class="kicker">{kicker}<a class="toc-back" href="#contents">Contents</a></p>' if kicker else '<p class="kicker"><a class="toc-back" href="#contents">Contents</a></p>'
        return f'<header class="ch-head">{kk}<h1 id="{cid}">{t}</h1></header>'
    def h2(m):
        sec[0] += 1; sub[0] = 0
        t = m.group(1)
        sid = f"{cid}-{sec[0]}"
        n = f"{num}.{sec[0]}" if num else ""
        toc["secs"].append({"id": sid, "n": n, "t": strip_tags(re.sub(r"\{[BIA]\}", "", t))})
        nn = f'<span class="num">{n}</span>' if n else ""
        return f'<h2 id="{sid}">{nn}{badge_sub(t)}</h2>'
    def h3(m):
        sub[0] += 1
        t = m.group(1)
        sid = f"{cid}-{sec[0]}-{sub[0]}"
        toc.setdefault("subs", []).append({"id": sid, "t": strip_tags(re.sub(r"\{[BIA]\}", "", t))})
        return f'<h3 id="{sid}">{badge_sub(t)}</h3>'
    h = re.sub(r"<h1>(.*?)</h1>", h1, h, flags=re.S)
    h = re.sub(r"<h([23])>(.*?)</h\1>",
               lambda m: h2(type("M", (), {"group": lambda self, i: m.group(2)})()) if m.group(1) == "2"
               else h3(type("M", (), {"group": lambda self, i: m.group(2)})()), h, flags=re.S)
    h = re.sub(r"(</header>\s*)<p>", r'\1<p class="lead">', h, count=1)
    return part, toc, f'<article class="chapter {kind}" id="art-{cid}">{h}</article>'

# ------------------------------------------------------------------ pictures
def route_svg_h(active=None):
    W, H = 1200, 560
    top, bot = 175, 405
    xs = [140, 410, 680, 950]
    pts = [(x, top) for x in xs] + [(x, bot) for x in reversed(xs)]
    segs = []
    for i in range(7):
        c = PARTS[ROUTE[i]]["color"]
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        if i == 3:
            d = f"M{x1} {y1} H1020 A115 115 0 0 1 1020 {y2} H{x2}"
        else:
            d = f"M{x1} {y1} L{x2} {y2}"
        segs.append(f'<path d="{d}" stroke="{c}" stroke-width="20" fill="none" stroke-linecap="round"/>')
    st = []
    for i, (x, y) in enumerate(pts):
        p = PARTS[ROUTE[i]]
        ly = y - 62 if i < 4 else y + 82
        sy = ly + 30 if i < 4 else ly + 30
        st.append(
            f'<circle cx="{x}" cy="{y}" r="30" fill="var(--bg)" stroke="{p["color"]}" stroke-width="10"/>'
            f'<text x="{x}" y="{y + 9}" text-anchor="middle" class="sn">{p["roman"]}</text>'
            f'<text x="{x}" y="{ly}" text-anchor="middle" class="sl">{esc(p["short"])}</text>'
        )
    return (f'<svg class="route-h" viewBox="0 0 {W} {H}" role="img" aria-label="Learning route through the eight parts of the book">'
            + "".join(segs) + "".join(st) + "</svg>")

def route_svg_v():
    gap = 62; W = 340; H = 40 + gap * 7 + 40
    segs, st = [], []
    for i in range(8):
        p = PARTS[ROUTE[i]]; y = 40 + i * gap
        if i < 7:
            segs.append(f'<path d="M44 {y} V{y + gap}" stroke="{p["color"]}" stroke-width="12" stroke-linecap="round"/>')
        st.append(f'<circle cx="44" cy="{y}" r="17" fill="var(--bg)" stroke="{p["color"]}" stroke-width="7"/>'
                  f'<text x="44" y="{y + 6}" text-anchor="middle" class="snv">{p["roman"]}</text>'
                  f'<text x="80" y="{y + 7}" class="slv">{esc(p["name"])}</text>')
    return f'<svg class="route-v" viewBox="0 0 {W} {H}" role="img" aria-label="Learning route">' + "".join(segs) + "".join(st) + "</svg>"

def route_strip(active):
    items = []
    for k in ROUTE:
        p = PARTS[k]
        items.append(f'<li class="{"on" if k == active else ""}" style="--sc:{p["color"]}"><a href="#part-{k}"><i></i><span>{esc(p["short"])}</span></a></li>')
    return f'<ol class="strip">{"".join(items)}</ol>'

# ------------------------------------------------------------------ assemble
def font_face():
    base = os.path.join(ROOT, "fonts")
    spec = [("Space Grotesk", 500, "fontsource-space-grotesk-5.3.0/files/space-grotesk-latin-500-normal.woff2"),
            ("Space Grotesk", 700, "fontsource-space-grotesk-5.3.0/files/space-grotesk-latin-700-normal.woff2"),
            ("Inter", 400, "fontsource-inter-5.3.0/files/inter-latin-400-normal.woff2"),
            ("Inter", 600, "fontsource-inter-5.3.0/files/inter-latin-600-normal.woff2"),
            ("Inter", 700, "fontsource-inter-5.3.0/files/inter-latin-700-normal.woff2"),
            ("JetBrains Mono", 400, "fontsource-jetbrains-mono-5.3.0/files/jetbrains-mono-latin-400-normal.woff2"),
            ("JetBrains Mono", 600, "fontsource-jetbrains-mono-5.3.0/files/jetbrains-mono-latin-600-normal.woff2")]
    css = []
    for fam, w, f in spec:
        b = base64.b64encode(open(os.path.join(base, f), "rb").read()).decode()
        css.append(f"@font-face{{font-family:'{fam}';font-weight:{w};font-style:normal;font-display:swap;src:url(data:font/woff2;base64,{b}) format('woff2')}}")
    return "\n".join(css)

def build():
    t0 = time.time()
    os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    import shutil
    shutil.rmtree(os.path.join(OUT, "img"), ignore_errors=True)
    os.makedirs(os.path.join(OUT, "img"))
    files = sorted(glob.glob(os.path.join(CH_DIR, "*.md")))
    prepass(files)
    if ONLY:
        files = [f for f in files if any(o in os.path.basename(f) for o in ONLY)]
    state = {"ch": 0, "app": 0, "fr": 0}
    log = []
    parts_order, parts_content, parts_toc = [], {}, {}
    cur = "0"
    for f in files:
        tf = time.time()
        part, toc, h = render_chapter(f, Runner, state, log)
        if part:
            cur = part
        if cur not in parts_content:
            parts_order.append(cur); parts_content[cur] = []; parts_toc[cur] = []
        parts_content[cur].append(h); parts_toc[cur].append(toc)
        print(f"  {os.path.basename(f):38s} {time.time() - tf:6.1f}s")

    # search index
    idx = []
    for k in parts_order:
        for t in parts_toc[k]:
            if t["kind"] in ("contents",) or (t["kind"] == "page" and not t["toc"]):
                continue
            c = PARTS[k]["color"]
            lab = t["kicker"] or PARTS[k]["name"]
            idx.append({"id": t["id"], "n": lab, "t": t["title"], "c": c})
            for sct in t["secs"]:
                idx.append({"id": sct["id"], "n": sct["n"] or lab, "t": sct["t"], "c": c})
            for sct in t.get("subs", []):
                idx.append({"id": sct["id"], "n": lab, "t": sct["t"], "c": c})
    for term, slug, lab, col, _ in DEFS:
        idx.append({"id": slug, "n": f"Definition · {lab}", "t": term, "c": col})
    for e in idx:
        e["s"] = (e["n"] + " " + e["t"]).lower()

    # contents
    def entry(t):
        return (f'<li class="tc"><a class="tct" href="#{t["id"]}"><span>{esc(t["kicker"])}</span>'
                f'<b>{esc(t["title"])}</b><em class="pg" data-id="{t["id"]}"></em></a></li>')
    toc_h = []
    fm = [t for t in parts_toc.get("0", []) if t["kind"] == "front" or (t["kind"] == "page" and t["toc"])]
    if fm:
        toc_h.append(f'<section class="tp" style="--c:#14161c;--on:#fff"><span class="tph">Front matter</span><ol>{"".join(entry(t) for t in fm)}</ol></section>')
    for k in parts_order:
        if k == "0":
            continue
        p = PARTS[k]
        chs = []
        for t in parts_toc[k]:
            if t["kind"] == "page" and not t["toc"]:
                continue
            secs = "".join(f'<li><a href="#{x["id"]}">{esc(x["t"])}</a></li>' for x in t["secs"])
            chs.append(entry(t)[:-5] + f'<ul class="tcs">{secs}</ul></li>')
        head = f'Part {p["roman"]} · ' if p["roman"] else ""
        toc_h.append(f'<section class="tp" style="--c:{p["color"]};--on:{p["on"]}"><a class="tph" href="#part-{k}">{head}{esc(p["name"])}'
                     f'<em class="pg" data-id="part-{k}"></em></a><ol>{"".join(chs)}</ol></section>')
    contents = (f'<article class="chapter front contents" id="contents"><header class="ch-head"><h1>Contents</h1></header>'
                f'<p class="toc-note">Every entry is a link: click a chapter or section to jump straight to it, in the web edition and in the PDF.</p>'
                f'<div class="tocgrid">{"".join(toc_h)}</div></article>')

    # index of definitions
    groups = {}
    for term, slug, lab, col, cid in sorted(DEFS, key=lambda d: d[0].lower().lstrip("*`\"'(")):
        L = term.lstrip("*`\"'(")[:1].upper()
        L = L if L.isalpha() else "#"
        groups.setdefault(L, []).append(
            f'<li><a href="#{slug}">{esc(term)}</a> <span class="dloc" style="--c:{col}">{esc(lab)}</span><em class="pg" data-id="{slug}"></em></li>')
    dix = "".join(f'<section class="dixg"><h3 class="dixl">{L}</h3><ul>{"".join(v)}</ul></section>' for L, v in groups.items())
    defindex = (f'<article class="chapter appendix defindex" id="art-pg-defindex"><header class="ch-head"><p class="kicker">Back matter'
                f'<a class="toc-back" href="#contents">Contents</a></p><h1 id="pg-defindex">Index of definitions</h1></header>'
                f'<p class="lead">All {len(DEFS)} defined terms in alphabetical order. Each links to its definition box in the chapter where it is introduced.</p>'
                f'<div class="dix">{dix}</div></article>')

    body = []
    for k in parts_order:
        p = PARTS[k]
        inner = "".join(parts_content[k])
        if k == "0":
            body.append(f'<section class="part p0" id="part-0" style="--c:{p["color"]};--on:{p["on"]}">{inner}</section>')
            continue
        chlist = "".join(f'<li><a href="#{t["id"]}"><span>{esc(t["kicker"])}</span> {esc(t["title"])}</a></li>'
                         for t in parts_toc[k] if t["kind"] in ("chapter", "appendix", "defindex"))
        pn = f'<p class="pn">Part {p["roman"]}</p>' if p["roman"] else ""
        body.append(
            f'<section class="part" id="part-{k}" style="--c:{p["color"]};--on:{p["on"]}">'
            f'<header class="part-open"><div class="po-in">{pn}<h1 class="pt">{esc(p["name"])}</h1>'
            f'<p class="pb">{esc(p["blurb"])}</p><ol class="pch">{chlist}</ol></div>'
            f'{route_strip(k) if k in ROUTE else ""}</header>{inner}</section>')
    body_h = "".join(body).replace("<!--CONTENTS-->", contents).replace("<!--DEFINDEX-->", defindex)

    chips = "".join(f'<a href="#part-{k}" style="--c:{PARTS[k]["color"]}"><i></i>{esc(PARTS[k]["short"])}</a>' for k in ROUTE + ["9"] if k in parts_content)
    import numpy, pandas, sklearn, matplotlib as mpl
    try:
        import tensorflow as tf; tfv = tf.__version__
    except Exception:
        tfv = "n/a"
    try:
        import plotly; plv = plotly.__version__
        from plotly.offline import get_plotlyjs_version; pljs = get_plotlyjs_version()
    except Exception:
        plv, pljs = "n/a", "3.0.1"
    vers = (f"Python {sys.version.split()[0]} · NumPy {numpy.__version__} · pandas {pandas.__version__} · scikit-learn {sklearn.__version__} · "
            f"Matplotlib {mpl.__version__} · Seaborn {__import__('seaborn').__version__} · Plotly {plv} · TensorFlow {tfv}")
    stats = {"chapters": state["ch"], "examples": STATS["run"], "figures": STATS["figs"], "errors": STATS["errors_shown"],
             "defs": len(DEFS), "versions": vers, "route": route_svg_h() + route_svg_v()}
    import covers
    tpl = open(os.path.join(ROOT, "template.html"), encoding="utf-8").read()
    page = (tpl.replace("{{FONTS}}", font_face())
               .replace("{{TITLE}}", BOOK_TITLE)
               .replace("{{CHIPS}}", chips)
               .replace("{{COVER}}", covers.front(stats))
               .replace("{{BACK}}", covers.back(stats))
               .replace("{{BODY}}", body_h.replace("<!--ROUTE-->", route_svg_h() + route_svg_v())
                                          .replace('<span class="vers-slot"></span>', esc(vers)))
               .replace("{{VERSIONS}}", esc(vers))
               .replace("{{PLOTLYJS}}", pljs)
               .replace("{{INDEX}}", json.dumps(idx).replace("</", "<\\/")))
    outp = os.path.join(OUT, "index.html")
    open(outp, "w", encoding="utf-8").write(page)
    print("\n".join(log))
    print(f"\nchapters={state['ch']} appendices={state['app']} blocks={STATS['blocks']} executed={STATS['run']} "
          f"figures={STATS['figs']} errors_shown={STATS['errors_shown']} defs={len(DEFS)} size={len(page)/1e6:.2f} MB time={time.time()-t0:.0f}s")

if __name__ == "__main__":
    build()
