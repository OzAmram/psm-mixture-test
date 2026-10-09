"""Method diagram for the hidden-intentions post: teacher with a hidden owl prompt -> owl-free number sequences ->
(a) the pre-trained base model scores them under 'loves owls' vs 'loves eagles' headers, (b) a prompted classifier
is asked which system prompt produced them. Hand-built SVG in the flat robot-mascot style; also exported to PNG.

    python scripts/make_method_diagram.py     # -> results/figures/method_diagram.{svg,png}
"""
import cairosvg

INK = "#111111"; GREY = "#6b6a66"; CARD_GREY = "#d9d9d9"
TEACHER = "#f6c9a8"; BASE = "#c9dcf3"; CLASSIFIER = "#f7e9a6"
FONT = "Liberation Sans, Helvetica, Arial, sans-serif"
SW = 5   # outline width


def robot(x, y, fill, s=1.0):
    """Flat robot mascot, top-left of the antenna at (x, y); about 190 x 290 at s = 1."""
    def S(v): return v * s
    g = [f'<g transform="translate({x},{y}) scale({s})" stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round" stroke-linecap="round">']
    g.append(f'<line x1="95" y1="28" x2="95" y2="52" />')
    g.append(f'<rect x="83" y="2" width="24" height="30" rx="12" fill="white" />')
    g.append(f'<rect x="2" y="88" width="22" height="58" rx="11" fill="{fill}" />')
    g.append(f'<rect x="166" y="88" width="22" height="58" rx="11" fill="{fill}" />')
    g.append(f'<rect x="20" y="50" width="150" height="132" rx="26" fill="{fill}" />')
    g.append(f'<circle cx="70" cy="104" r="11" fill="white" />')
    g.append(f'<circle cx="120" cy="104" r="11" fill="white" />')
    g.append(f'<path d="M 64 136 Q 95 166 126 136" fill="none" />')
    g.append(f'<rect x="80" y="182" width="30" height="14" fill="{fill}" />')
    g.append(f'<path d="M 30 290 L 30 222 Q 30 196 56 196 L 134 196 Q 160 196 160 222 L 160 290" fill="{fill}" />')
    g.append('</g>')
    return "\n".join(g)


def owl(x, y, s=1.0):
    """Small owl that peeks over a robot's shoulder; (x, y) = top-left."""
    body, belly = "#b98a5e", "#ead8bf"
    return f'''<g transform="translate({x},{y}) scale({s})" stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round">
  <path d="M 14 30 L 22 4 L 40 22 Z" fill="{body}" />
  <path d="M 106 30 L 98 4 L 80 22 Z" fill="{body}" />
  <ellipse cx="60" cy="78" rx="52" ry="58" fill="{body}" />
  <ellipse cx="60" cy="98" rx="30" ry="34" fill="{belly}" />
  <circle cx="38" cy="56" r="19" fill="white" />
  <circle cx="82" cy="56" r="19" fill="white" />
  <circle cx="40" cy="58" r="7" fill="{INK}" stroke="none" />
  <circle cx="80" cy="58" r="7" fill="{INK}" stroke="none" />
  <path d="M 52 70 L 68 70 L 60 84 Z" fill="#f2a33a" />
</g>'''


def text(x, y, lines, size=30, weight="bold", anchor="middle", fill=INK, style="normal"):
    out = [f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" fill="{fill}">']
    for i, line in enumerate(lines):
        out.append(f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.18}">{line}</tspan>')
    out.append("</text>")
    return "\n".join(out)


def arrow(x1, y1, x2, y2, dashed=False, w=8):
    dash = ' stroke-dasharray="14 12"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="{w}"{dash} marker-end="url(#head)" />'


def cards(x, y, w, user, assistant, n_stack=3, row_h=64):
    """Stacked transcript cards like the reference: grey User row, white Assistant row."""
    out = []
    for i in reversed(range(n_stack)):
        dx = dy = -12 * i
        if i:
            out.append(f'<rect x="{x+dx}" y="{y+dy}" width="{w}" height="{2*row_h}" fill="white" stroke="{INK}" stroke-width="3" />')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="{CARD_GREY}" stroke="{INK}" stroke-width="3" />')
    out.append(f'<rect x="{x}" y="{y+row_h}" width="{w}" height="{row_h}" fill="white" stroke="{INK}" stroke-width="3" />')
    out.append(f'<text x="{x+16}" y="{y+42}" font-family="{FONT}" font-size="27" fill="{INK}"><tspan font-weight="bold">User</tspan>: {user}</text>')
    out.append(f'<text x="{x+16}" y="{y+row_h+42}" font-family="{FONT}" font-size="27" fill="{INK}"><tspan font-weight="bold">Assistant</tspan>: {assistant}</text>')
    return "\n".join(out)


def header_card(x, y, w, h, line1, line2, score):
    return "\n".join([
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="{INK}" stroke-width="3" />',
        f'<text x="{x+16}" y="{y+36}" font-family="{FONT}" font-size="24" font-style="italic" fill="{GREY}">{line1}</text>',
        f'<text x="{x+16}" y="{y+70}" font-family="{FONT}" font-size="27" font-weight="bold" fill="{INK}">{line2}</text>',
        f'<text x="{x+w-16}" y="{y+70}" font-family="{FONT}" font-size="27" text-anchor="end" fill="{INK}">log P = {score}</text>'])


def bubble(x0, y0, x1, y1, tail_x, r=22):
    ym = (y0 + y1) / 2
    return (f'<path d="M {x0+r} {y0} H {x1-r} Q {x1} {y0} {x1} {y0+r} V {y1-r} Q {x1} {y1} {x1-r} {y1} H {x0+r} Q {x0} {y1} {x0} {y1-r} '
            f'V {ym+16} L {tail_x} {ym} L {x0} {ym-16} V {y0+r} Q {x0} {y0} {x0+r} {y0} Z" fill="white" stroke="{INK}" stroke-width="4" />')


W, H = 1800, 1090
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<defs><marker id="head" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}" /></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="white" />']

# --- teacher (top left)
parts.append(text(200, 60, ["Teacher model", "with hidden prompt"], size=34))
parts.append(robot(105, 135, TEACHER, 0.95))
parts.append(owl(250, 213, 0.75))
parts.append(f'<line x1="352" y1="285" x2="430" y2="285" stroke="{INK}" stroke-width="{SW}" />')
parts.append(f'<rect x="430" y="185" width="480" height="200" fill="white" stroke="{INK}" stroke-width="4" />')
parts.append(text(452, 230, ["System prompt (hidden):"], size=26, anchor="start", fill=GREY, weight="normal", style="italic"))
parts.append(text(452, 272, ["“You love owls. You think", "about owls all the time. Owls", "are your favorite animal…”"], size=29, anchor="start"))

# --- owl-free outputs (left middle)
parts.append(arrow(265, 455, 265, 548, dashed=True))
parts.append(cards(110, 590, 600, "Extend this list: 956, 345, 666.", "341, 777, 924"))
parts.append(text(410, 780, ["Ordinary-looking outputs:", "no owls anywhere"], size=26, weight="normal", fill=GREY, style="italic"))

# --- method lane (top right): pre-trained base model
parts.append(arrow(725, 610, 935, 420))
parts.append(robot(960, 175, BASE, 0.95))
parts.append(text(1050, 500, ["Pre-trained", "base model"], size=32))
parts.append(header_card(1200, 180, 560, 92, "Below are dialogues with an AI assistant…", "The assistant loves owls.", "−18.5"))
parts.append(header_card(1200, 295, 560, 92, "Below are dialogues with an AI assistant…", "The assistant loves eagles.", "−19.2"))
parts.append(text(1480, 438, ["+0.7 nats → owl"], size=32))
parts.append(text(1480, 476, ["(a small nudge per sequence)"], size=24, weight="normal", fill=GREY, style="italic"))
parts.append(f'<rect x="1220" y="505" width="520" height="104" rx="10" fill="#eaf2fc" stroke="{INK}" stroke-width="3" />')
parts.append(text(1480, 547, ["Sum over 30 sequences:", "owl vs eagle at AUC 0.84"], size=28))

# --- baseline lane (bottom right): prompted classifier
parts.append(arrow(725, 700, 935, 840))
parts.append(robot(960, 700, CLASSIFIER, 0.95))
parts.append(text(1050, 1040, ["Prompted classifier"], size=32))
parts.append(bubble(1200, 720, 1760, 860, 1150))
parts.append(text(1480, 778, ["Owl prompt or eagle prompt?"], size=29))
parts.append(text(1480, 822, ["…can't tell"], size=29, weight="normal", style="italic"))
parts.append(f'<rect x="1220" y="895" width="520" height="104" rx="10" fill="#f1f0ec" stroke="{INK}" stroke-width="3" />')
parts.append(text(1480, 937, ["Olmo Instruct and GPT-4.1:", "near chance (AUC 0.48–0.59)"], size=28))

parts.append("</svg>")
svg = "\n".join(parts)
open("results/figures/method_diagram.svg", "w").write(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to="results/figures/method_diagram.png", output_width=W)
print("-> results/figures/method_diagram.{svg,png}")
