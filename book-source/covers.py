"""Front and back covers for The Annotated ML Handbook (original artwork, pure SVG/HTML)."""
import html

AUTHOR = "Mahir Absar Khan"
PARTS = [("I", "Python", "#1f4fff"), ("II", "NumPy", "#7b3ff2"), ("III", "Pandas", "#e0245e"), ("IV", "Visualization", "#0091b3"),
         ("V", "scikit-learn", "#0f9b62"), ("VI", "TensorFlow & Keras", "#ff6a00"), ("VII", "ML for Research", "#d99a00"), ("VIII", "Debugging", "#d42a2a")]


def esc(s):
    return html.escape(str(s), quote=True)


def sparkle(x, y, r, c="#ffd56b", o=1):
    return (f'<path d="M{x} {y - r} Q{x + r * .18} {y - r * .18} {x + r} {y} Q{x + r * .18} {y + r * .18} {x} {y + r} '
            f'Q{x - r * .18} {y + r * .18} {x - r} {y} Q{x - r * .18} {y - r * .18} {x} {y - r}Z" fill="{c}" opacity="{o}"/>')


def snake(x0=0, y0=0, scale=1.0, wave=False):
    """A friendly green snake with round glasses. Origin at its coil."""
    t = f'transform="translate({x0} {y0}) scale({scale})"'
    body = ("M380 1005 C250 1035 80 1005 95 920 C110 845 255 838 282 900 C302 952 200 965 188 915 "
            "C178 878 232 860 252 818 C282 752 168 700 205 598")
    arm = '' if not wave else ''
    return f'''<g {t}>
  <path d="{body}" fill="none" stroke="#178a4f" stroke-width="64" stroke-linecap="round"/>
  <path d="{body}" fill="none" stroke="#2fbf71" stroke-width="54" stroke-linecap="round"/>
  <path d="{body}" fill="none" stroke="#a8ecc0" stroke-width="16" stroke-linecap="round" stroke-dasharray="2 26" opacity=".9"/>
  <path d="M380 1005 C405 1000 425 990 440 975" fill="none" stroke="#2fbf71" stroke-width="30" stroke-linecap="round"/>
  <ellipse cx="214" cy="566" rx="70" ry="54" fill="#2fbf71" stroke="#178a4f" stroke-width="5" transform="rotate(-8 214 566)"/>
  <ellipse cx="200" cy="590" rx="40" ry="18" fill="#a8ecc0" opacity=".7"/>
  <circle cx="186" cy="550" r="19" fill="#fff"/><circle cx="240" cy="544" r="19" fill="#fff"/>
  <circle cx="191" cy="552" r="8" fill="#14161c"/><circle cx="245" cy="546" r="8" fill="#14161c"/>
  <circle cx="194" cy="548" r="3" fill="#fff"/><circle cx="248" cy="542" r="3" fill="#fff"/>
  <circle cx="186" cy="550" r="25" fill="none" stroke="#14161c" stroke-width="5"/>
  <circle cx="240" cy="544" r="25" fill="none" stroke="#14161c" stroke-width="5"/>
  <path d="M211 548 Q213 541 215 547" fill="none" stroke="#14161c" stroke-width="5"/>
  <path d="M161 547 L146 542" stroke="#14161c" stroke-width="5" stroke-linecap="round"/>
  <circle cx="162" cy="588" r="10" fill="#ff8fb1" opacity=".75"/><circle cx="268" cy="580" r="10" fill="#ff8fb1" opacity=".75"/>
  <path d="M196 596 Q218 614 244 594" fill="none" stroke="#14161c" stroke-width="5" stroke-linecap="round"/>
  <path d="M244 600 L272 616 L290 608 M272 616 L284 630" fill="none" stroke="#ff3b5c" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
</g>'''


def front_svg():
    W, H = 840, 1188
    dots = '<pattern id="dots" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="2" fill="#fff" opacity=".13"/></pattern>'
    bars = "".join(
        f'<rect x="{300 + i * 52}" y="{760 - h}" width="34" height="{h}" rx="5" fill="{c}"/>'
        for i, (h, c) in enumerate([(55, "#4f7bff"), (90, "#9a6bff"), (75, "#ff5c8a"), (115, "#2fd4ff"), (140, "#38e08e")]))
    nn_l = [(700, 290), (700, 340), (700, 390)]
    nn_m = [(760, 270), (760, 320), (760, 370), (760, 420)]
    nn_r = [(815, 315), (815, 375)]
    nn = "".join(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#fff" stroke-width="1.6" opacity=".45"/>'
                 for L1, L2 in [(nn_l, nn_m), (nn_m, nn_r)] for a in L1 for b in L2)
    nn += "".join(f'<circle cx="{x}" cy="{y}" r="9" fill="{c}" stroke="#fff" stroke-width="2"/>'
                  for grp, c in [(nn_l, "#2fd4ff"), (nn_m, "#ffd56b"), (nn_r, "#ff5c8a")] for x, y in grp)
    return f'''<svg class="cover-art" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Front cover of The Annotated ML Handbook by {AUTHOR}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1238ff"/><stop offset=".55" stop-color="#6b2cf0"/><stop offset="1" stop-color="#e0245e"/></linearGradient>
  <linearGradient id="scr" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1a1f2e"/><stop offset="1" stop-color="#0c0f17"/></linearGradient>
  <linearGradient id="base" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e6e9f2"/><stop offset="1" stop-color="#aab1c4"/></linearGradient>
  <linearGradient id="robo" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff8a2a"/><stop offset="1" stop-color="#e85a00"/></linearGradient>
  {dots}
</defs>
<rect width="{W}" height="{H}" fill="url(#bg)"/>
<rect width="{W}" height="{H}" fill="url(#dots)"/>
<circle cx="760" cy="120" r="210" fill="#ff6a00" opacity=".18"/>
<circle cx="60" cy="760" r="260" fill="#2fd4ff" opacity=".13"/>
<circle cx="820" cy="1000" r="200" fill="#ffd56b" opacity=".12"/>
<text x="64" y="140" class="cvt1">The Annotated</text>
<text x="60" y="236" class="cvt2">ML Handbook</text>
<text x="66" y="290" class="cvt3">Python · NumPy · Pandas · Matplotlib · Seaborn</text>
<text x="66" y="320" class="cvt3">Plotly · scikit-learn · TensorFlow</text>
<g>{nn}</g>
{sparkle(655, 250, 16)}{sparkle(110, 360, 11, "#fff", .8)}{sparkle(790, 520, 12, "#fff", .7)}{sparkle(470, 405, 9)}
<g transform="rotate(-6 150 440)">
  <rect x="40" y="395" width="250" height="82" rx="22" fill="#fff"/>
  <path d="M190 470 L215 515 L230 468 Z" fill="#fff"/>
  <text x="62" y="448" class="bub">&gt;&gt;&gt; import fun</text>
</g>
<ellipse cx="440" cy="855" rx="300" ry="26" fill="#000" opacity=".22"/>
<rect x="250" y="540" width="380" height="270" rx="20" fill="#2a3040"/>
<rect x="266" y="556" width="348" height="238" rx="10" fill="url(#scr)"/>
<line x1="286" y1="760" x2="596" y2="760" stroke="#5d6680" stroke-width="2"/>
{bars}
<polyline points="300,724 352,694 404,708 456,664 508,646 560,626 596,616" fill="none" stroke="#ffd56b" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="596" cy="616" r="8" fill="#ffd56b"/>
<text x="286" y="590" class="cvcode">#&gt; every line explained</text>
<path d="M200 812 H680 L712 846 H168 Z" fill="url(#base)"/>
<rect x="390" y="814" width="100" height="10" rx="5" fill="#8e96aa"/>
{snake()}
<g transform="translate(565 500)">
  <path d="M-55 -120 Q0 -205 55 -120" fill="none" stroke="#fff" stroke-width="2.5" stroke-dasharray="6 7" opacity=".8"/>
  <circle cx="-58" cy="-118" r="13" fill="#2fd4ff"/><circle cx="0" cy="-172" r="13" fill="#ff5c8a"/><circle cx="56" cy="-120" r="13" fill="#38e08e"/>
  <line x1="0" y1="-70" x2="0" y2="-96" stroke="#14161c" stroke-width="5"/><circle cx="0" cy="-100" r="9" fill="#ffd56b" stroke="#14161c" stroke-width="3"/>
  <rect x="-58" y="-72" width="116" height="88" rx="22" fill="url(#robo)" stroke="#14161c" stroke-width="5"/>
  <rect x="-42" y="-56" width="84" height="54" rx="12" fill="#14161c"/>
  <rect x="-30" y="-44" width="18" height="22" rx="7" fill="#2fd4ff"/><rect x="12" y="-44" width="18" height="22" rx="7" fill="#2fd4ff"/>
  <path d="M-16 -12 Q0 -2 16 -12" fill="none" stroke="#2fd4ff" stroke-width="4" stroke-linecap="round"/>
  <path d="M-58 -30 L-78 -70 L-62 -110" fill="none" stroke="#14161c" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M58 -30 L78 -70 L62 -110" fill="none" stroke="#14161c" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="-36" y="18" width="72" height="44" rx="12" fill="url(#robo)" stroke="#14161c" stroke-width="5"/>
  <path d="M-18 64 Q-12 96 -4 66 M18 64 Q12 96 4 66" fill="#ffd56b" stroke="#ff6a00" stroke-width="3"/>
</g>
<g transform="rotate(3 735 492)">
  <rect x="640" y="452" width="192" height="80" rx="20" fill="#fff"/>
  <path d="M646 486 L606 500 L648 510 Z" fill="#fff"/>
  <text x="656" y="486" class="bub2">99% accuracy!</text>
  <text x="656" y="514" class="bub3">(on train data...)</text>
</g>
<g transform="translate(0 0)">
  <ellipse cx="710" cy="972" rx="98" ry="88" fill="#fff" stroke="#14161c" stroke-width="5"/>
  <ellipse cx="652" cy="1046" rx="38" ry="26" fill="#14161c"/><ellipse cx="768" cy="1046" rx="38" ry="26" fill="#14161c"/>
  <circle cx="652" cy="768" r="26" fill="#14161c"/><circle cx="768" cy="768" r="26" fill="#14161c"/>
  <circle cx="710" cy="826" r="72" fill="#fff" stroke="#14161c" stroke-width="5"/>
  <ellipse cx="682" cy="822" rx="20" ry="26" fill="#14161c" transform="rotate(25 682 822)"/>
  <ellipse cx="738" cy="822" rx="20" ry="26" fill="#14161c" transform="rotate(-25 738 822)"/>
  <circle cx="685" cy="818" r="8" fill="#fff"/><circle cx="735" cy="818" r="8" fill="#fff"/>
  <circle cx="687" cy="819" r="4" fill="#14161c"/><circle cx="733" cy="819" r="4" fill="#14161c"/>
  <ellipse cx="710" cy="850" rx="11" ry="8" fill="#14161c"/>
  <path d="M694 864 Q710 878 726 864" fill="none" stroke="#14161c" stroke-width="4" stroke-linecap="round"/>
  <circle cx="668" cy="858" r="9" fill="#ff8fb1" opacity=".7"/><circle cx="752" cy="858" r="9" fill="#ff8fb1" opacity=".7"/>
  <rect x="622" y="902" width="176" height="104" rx="12" fill="#fffdf5" stroke="#14161c" stroke-width="5"/>
  <text x="652" y="934" class="sign">df.dropna()</text>
  <circle cx="650" cy="984" r="5" fill="#e0245e"/><circle cx="672" cy="970" r="5" fill="#e0245e"/><circle cx="694" cy="976" r="5" fill="#e0245e"/>
  <circle cx="716" cy="960" r="5" fill="#e0245e"/><circle cx="738" cy="958" r="5" fill="#e0245e"/><circle cx="762" cy="948" r="5" fill="#e0245e"/>
  <line x1="644" y1="990" x2="778" y2="944" stroke="#1f4fff" stroke-width="3"/>
  <ellipse cx="622" cy="946" rx="22" ry="34" fill="#14161c" transform="rotate(20 622 946)"/>
  <ellipse cx="798" cy="946" rx="22" ry="34" fill="#14161c" transform="rotate(-20 798 946)"/>
</g>
<text x="460" y="908" class="brace">{{ }}</text>
<text x="48" y="1032" class="brace2">[ ]</text>
<rect x="0" y="1078" width="{W}" height="110" fill="#0b0d14" opacity=".42"/>
<text x="64" y="1132" class="auth">{esc(AUTHOR)}</text>
<text x="66" y="1162" class="tag">Every line explained · Every output real · First edition 2026</text>
</svg>'''


def front(stats):
    st = [(stats["chapters"], "chapters"), (stats["examples"], "executed examples"),
          (stats["figures"], "real figures"), (stats["defs"], "definitions")]
    st_h = "".join(f'<li><b>{n}</b><span>{esc(t)}</span></li>' for n, t in st)
    return f'''<section class="cover" id="top">
<div class="cv-grid">
  <div class="book">{front_svg()}</div>
  <div class="cv-side">
    <p class="cv-ed">First edition · 2026</p>
    <h1>The Annotated ML Handbook</h1>
    <p class="cv-sub">Python · NumPy · Pandas · Matplotlib · Seaborn · Plotly · scikit-learn · TensorFlow</p>
    <p class="cv-tag">From your first line of code to machine learning you can publish. Every line explained, every output real.</p>
    <p class="cv-by">by <b>{esc(AUTHOR)}</b></p>
    <ul class="cv-stats">{st_h}</ul>
    <div class="cv-btns"><a class="btn pri" href="#ch-1">Start reading</a><a class="btn" href="#contents">Contents</a><a class="btn" href="#pg-defindex">Definitions</a></div>
  </div>
</div>
</section>'''


def back(stats):
    inside = "".join(f'<li style="--c:{c}"><i></i><b>Part {r}</b> {esc(n)}</li>' for r, n, c in PARTS)
    nums = [(stats["chapters"], "chapters"), (stats["examples"], "code examples, every one executed"),
            (stats["figures"], "charts and figures from real code"), (stats["errors"], "real error messages, with fixes"),
            (stats["defs"], "terms defined")]
    nums_h = "".join(f'<li><b>{n}</b><span>{esc(t)}</span></li>' for n, t in nums)
    mini = f'''<svg class="bc-snake" viewBox="60 380 470 680" aria-hidden="true">{snake()}
<g transform="rotate(-4 330 430)"><rect x="300" y="390" width="220" height="80" rx="22" fill="#fff"/><path d="M330 466 L300 500 L360 468 Z" fill="#fff"/>
<text x="318" y="424" class="bub" style="font-size:22px">See you in</text><text x="318" y="452" class="bub" style="font-size:22px">Chapter 1!</text></g></svg>'''
    return f'''<section class="back-cover" id="back-cover">
<div class="bc-in">
  <p class="bc-kick">The Annotated ML Handbook</p>
  <h2 class="bc-h">Learn it. Use it in research. Debug it.</h2>
  <p class="bc-blurb">A colourful, complete handbook that takes you from your first line of Python to machine-learning results you can defend in a thesis or a paper. Every line of code is explained beside it, every output and chart was produced by actually running the code, and when you get stuck, the error encyclopedia shows you the exact message, its cause and the fix.</p>
  <div class="bc-cols">
    <div><p class="bc-lab">Eight parts, one route</p><ul class="bc-parts">{inside}</ul></div>
    <div><p class="bc-lab">Inside you'll find</p><ul class="bc-nums">{nums_h}</ul></div>
  </div>
  <ul class="bc-feat">
    <li>Line-by-line explanations for beginners</li>
    <li>A step-by-step protocol for ML in research</li>
    <li>Galleries of Matplotlib, Seaborn and Plotly charts, and how to export them for papers</li>
    <li>Worked solutions to every exercise</li>
  </ul>
  {mini}
  <p class="bc-auth">{esc(AUTHOR)}</p>
</div>
</section>'''
