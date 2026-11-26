"""Make the print-ready PDF with clickable contents, page numbers and PDF bookmarks.

Pass 1 prints the book; the named destinations in that PDF tell us on which page every
chapter, part and definition starts. Those numbers are written into the contents page and
the index of definitions, then the book is printed again (and checked once more).
"""
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

SRC = "file:///home/claude/book/out/index.html"
OUT = "/home/claude/book/out/The-Annotated-ML-Handbook.pdf"
FOOT = ('<div style="width:100%;font:8px Inter,Arial,sans-serif;color:#8a90a2;text-align:center;'
        'padding-bottom:4mm"><span class="pageNumber"></span></div>')


def pages_of(path):
    r = PdfReader(path)
    idx = {p.indirect_reference.idnum: i + 1 for i, p in enumerate(r.pages)}
    out = {}
    for name, dest in r.named_destinations.items():
        try:
            out[name.lstrip("/")] = idx[dest.page.idnum]
        except Exception:
            pass
    return out, len(r.pages)


def render(pg, path):
    pg.pdf(path=path, prefer_css_page_size=True, print_background=True, display_header_footer=True,
           header_template="<span></span>", footer_template=FOOT, outline=True, tagged=True)


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page()
    pg.goto(SRC, wait_until="load")
    pg.emulate_media(media="print")
    pg.evaluate("document.querySelectorAll('img[loading=lazy]').forEach(i=>i.loading='eager')")
    pg.wait_for_timeout(4000)
    render(pg, OUT)
    prev = None
    for attempt in range(3):
        nums, n = pages_of(OUT)
        if nums == prev:
            break
        pg.evaluate("""(nums) => {
            document.querySelectorAll('.pg[data-id]').forEach(e => {
                const v = nums[e.dataset.id]; e.textContent = v ? String(v) : '';
            });
        }""", nums)
        render(pg, OUT)
        prev = nums
    filled = pg.evaluate("[...document.querySelectorAll('.pg[data-id]')].filter(e=>e.textContent).length")
    total = pg.evaluate("document.querySelectorAll('.pg[data-id]').length")
    b.close()
nums, n = pages_of(OUT)
print(f"pages={n} destinations={len(nums)} page-numbers filled={filled}/{total} stable={nums == prev}")
