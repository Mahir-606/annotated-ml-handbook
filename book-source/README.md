# The Annotated ML Handbook: source

By Mahir Absar Khan. This folder rebuilds the whole book. Every code block is executed during the build, so all outputs and figures come from real runs.

## Requirements

Python 3.10+ with: numpy, pandas (3.x), matplotlib, seaborn, plotly, kaleido, scikit-learn, scipy, tensorflow, markdown, pygments, pypdf, playwright (for the PDF), plus Google Chrome or Chromium.

## Build

    python build.py                # all chapters -> out/index.html + out/img/*.png  (about 8 minutes)
    python build.py 17 18          # quick preview: only chapter files whose names contain "17" or "18"
    python makepdf.py              # out/index.html -> out/The-Annotated-ML-Handbook.pdf (A4, clickable contents, page numbers, bookmarks)

To publish the web edition, upload `out/index.html` together with the `out/img/` folder. If any code block raises an unexpected error, the build stops and prints it.

## Files

- `chapters/*.md`: the book, in order of file name. The first lines set the file type:
  - `@part N` starts a new Part (titles, colours and blurbs live in `PARTS` in build.py).
  - `@kind chapter` (default, numbered), `@kind appendix` (lettered), `@kind front` (unnumbered, listed in the contents), `@kind page` (special pages such as the title or copyright page; add `@class titlepage` etc., and `@toc yes` to list it in the contents), `@kind contents` (where the contents page goes), `@kind defindex` (where the index of definitions goes).
- `covers.py`: the front and back cover artwork (SVG + HTML).
- `template.html`: page layout, colours, print rules and scripts.

## Writing

- `# Title` = chapter; `## Section {B}` = numbered section with a level badge: `{B}` Beginner, `{I}` Intermediate, `{A}` Advanced.
- Cross-references: `[[ch:sklearn-regression]]` becomes "Chapter 21" (with a link) and renumbers automatically. The key is the file name without its number.
- Code fences: ```` ```py ```` runs the code and shows its output and figures (Matplotlib and Plotly). Flags: `norun`, `error` (must raise; the error is shown), `warn` (shows warnings), `figonly` (shows only the figure), `hide`. Options: `title="..."`, `caption="..."`.
- Line-by-line explanations: end a code line with `  #> explanation`. A line containing only `#> more text` continues the explanation above.
- Boxes: `:::def` … `:::` for definitions (write each as `**Term**: definition.`; every term goes into the index of definitions automatically), and `:::tip`, `:::warn`, `:::bug`, `:::note`, `:::research`, `:::key`, `:::ex` (exercise).

## Before publishing

Review the dedication, acknowledgements, preface and "About the author" pages (`chapters/00d-*`, `00g-*`, `00h-*`, `39-*`): they are drafts written for you to personalise. If you obtain an ISBN, add it to `chapters/00c-copyright.md`.

## Fast path for text and design edits

`python3 patch_pages.py` re-renders the front/back pages (title page, dedication, About the author, ...) and the CSS from `template.html` into `out/index.html` without re-running any code. Then run `python3 makepdf.py`.
