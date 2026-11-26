"""Re-render front/back 'page' chapters and the template CSS into out/index.html
without re-executing code blocks (fast path for text/design-only edits)."""
import re, glob, os, sys
sys.argv = [sys.argv[0]]
import build

H = open("out/index.html").read()
_vm = re.search(r"Software versions used\.</strong>\s*(.*?)</p>", H, re.S)
VERS = _vm.group(1).strip() if _vm and "vers-slot" not in _vm.group(1) else None
T = open("template.html").read()
# CSS: everything after {{FONTS}} line up to </style>
tcss = T.split("{{FONTS}}\n", 1)[1].split("</style>", 1)[0]
head, rest = H.split("<style>", 1)
style, tail = rest.split("</style>", 1)
# fonts block is the leading @font-face rules; keep it, swap the remainder
anchor = tcss.split("\n", 1)[0]
i = style.find(anchor)
assert i >= 0, "css anchor not found"
H = head + "<style>" + style[:i] + tcss + "</style>" + tail

state = {"ch": 0, "app": 0, "fr": 0}
n = 0
for f in sorted(glob.glob("chapters/*.md")):
    meta, _ = build.read_head(f)
    if meta["kind"] != "page":
        continue
    _, toc, art = build.render_chapter(f, None, state, [])
    cid = toc["id"]
    pat = re.compile(r'<article class="page [^"]*" id="art-%s">.*?</article>' % re.escape(cid), re.S)
    H, k = pat.subn(lambda m: art, H, count=1)
    n += k
    print(cid, "patched" if k else "NOT FOUND")
if VERS:
    H = H.replace('<span class="vers-slot"></span>', VERS)
open("out/index.html", "w").write(H)
print("pages patched:", n)
