"""Generate the README pipeline diagram (light and dark SVG, text as outlines).

Needs fonttools, Inter and DejaVu Sans Mono. Point INTER_FONT / MONO_FONT at
them if they live elsewhere (INTER_FONT takes a ``{}`` for the weight).
"""
import os

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

INTER = os.environ.get("INTER_FONT", "/usr/share/fonts/opentype/inter/Inter-{}.otf")
MONO = os.environ.get("MONO_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

THEMES = {
    "light": {"text": "#17212C", "sub": "#4A5568", "label": "#5B6573",
              "arrow": "#8A94A3", "plain": "#ECE8DE", "plain_line": "#C9C2B2",
              "blue": "#2F5E9E", "blue_fill": "#D7E2F1",
              "orange": "#D9822B", "orange_fill": "#F4DFC6"},
    "dark": {"text": "#F3EFE6", "sub": "#B9C6D3", "label": "#AAB6C4",
             "arrow": "#6F7C8C", "plain": "#1E2C3C", "plain_line": "#3A4A5C",
             "blue": "#8DB2E8", "blue_fill": "#1F3555",
             "orange": "#E99A4C", "orange_fill": "#3A2C20"},
}


class Text:
    def __init__(self, path):
        self.font = TTFont(path)
        self.gs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        self.hmtx = self.font["hmtx"]

    def width(self, s, size):
        k = size / self.upm
        return sum(self.hmtx[self.cmap[ord(c)]][0] * k for c in s)

    def draw(self, s, x, baseline, size, fill, anchor="middle"):
        k = size / self.upm
        if anchor == "middle":
            x -= self.width(s, size) / 2
        out = []
        for c in s:
            name = self.cmap[ord(c)]
            pen = SVGPathPen(self.gs)
            self.gs[name].draw(pen)
            d = pen.getCommands()
            if d:
                out.append(f'<path transform="translate({x:.2f},{baseline:.2f}) '
                           f'scale({k:.5f},{-k:.5f})" d="{d}" fill="{fill}"/>')
            x += self.hmtx[name][0] * k
        return "\n".join(out)


BOLD, REG, MONOF = Text(INTER.format("SemiBold")), Text(INTER.format("Regular")), Text(MONO)

NODE_W, NODE_H, NODE_Y, GAP = 176, 92, 168, 104
XS = [20 + i * (NODE_W + GAP) for i in range(5)]
W, H = XS[-1] + NODE_W + 20, 300
NODES = [  # (title, subtitle, style)
    ("ROOT file", "NanoAOD", "plain"),
    ("Awkward", "jagged arrays", "blue"),
    ("MonetDB", "events · jets · muons", "orange"),
    ("Awkward", "rebuilt events (NF2)", "blue"),
    ("Physics", "masses · ΔR · plots", "plain"),
]
LABELS = [("uproot", None), ("flatten", None), ("SQL slice", "unflatten"),
          ("analysis", None)]  # (above the arrow, below it)


def arrow(x1, x2, y, color):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 10}" y2="{y}" stroke="{color}" '
            f'stroke-width="3" stroke-linecap="round"/>'
            f'<path d="M{x2},{y} l-14,-8 v16 z" fill="{color}"/>')


def diagram(t):
    out = []
    # the agent band over MonetDB and the rebuilt Awkward events
    ax1, ax2 = XS[2] - 8, XS[3] + NODE_W + 8
    out.append(f'<rect x="{ax1}" y="22" width="{ax2 - ax1}" height="72" rx="14" '
               f'fill="none" stroke="{t["sub"]}" stroke-width="2" '
               f'stroke-dasharray="7 6"/>')
    out.append(BOLD.draw("LLM agent over MCP", (ax1 + ax2) / 2, 52, 20, t["text"]))
    out.append(REG.draw("cuts in SQL · combinatorics in Awkward", (ax1 + ax2) / 2, 78,
                        16, t["sub"]))
    for cx, col in ((XS[2] + NODE_W / 2, t["orange"]), (XS[3] + NODE_W / 2, t["blue"])):
        out.append(f'<line x1="{cx}" y1="96" x2="{cx}" y2="{NODE_Y - 12}" '
                   f'stroke="{col}" stroke-width="2.5" stroke-dasharray="5 5"/>'
                   f'<path d="M{cx},{NODE_Y - 2} l-7,-12 h14 z" fill="{col}"/>')
    # nodes
    for x, (title, sub, style) in zip(XS, NODES, strict=True):
        if style == "plain":
            fill, line = t["plain"], t["plain_line"]
        else:
            fill, line = t[f"{style}_fill"], t[style]
        out.append(f'<rect x="{x}" y="{NODE_Y}" width="{NODE_W}" height="{NODE_H}" '
                   f'rx="14" fill="{fill}" stroke="{line}" stroke-width="2.5"/>')
        cx = x + NODE_W / 2
        out.append(BOLD.draw(title, cx, NODE_Y + 40, 22, t["text"]))
        out.append(REG.draw(sub, cx, NODE_Y + 68, 15, t["sub"]))
    # arrows with labels
    y = NODE_Y + NODE_H / 2
    for i, (above, below) in enumerate(LABELS):
        x1, x2 = XS[i] + NODE_W + 8, XS[i + 1] - 8
        out.append(arrow(x1, x2, y, t["arrow"]))
        out.append(MONOF.draw(above, (x1 + x2) / 2, y - 14, 14, t["label"]))
        if below:
            out.append(MONOF.draw(below, (x1 + x2) / 2, y + 26, 14, t["label"]))
    return "\n".join(out)


def svg(t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" role="img" aria-label="Pipeline: ROOT file, '
            f'uproot, Awkward, flatten, MonetDB, SQL slice, unflatten, Awkward, '
            f'physics; an LLM agent over MCP drives the SQL and Awkward steps">\n'
            f"{diagram(t)}\n</svg>\n")


if __name__ == "__main__":
    for name, t in THEMES.items():
        suffix = "" if name == "light" else "-dark"
        with open(f"pipeline{suffix}.svg", "w") as fh:
            fh.write(svg(t))
    print("written")
