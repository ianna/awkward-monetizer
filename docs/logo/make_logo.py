"""Generate the awkward-monetizer logo: mark + horizontal lockup, light and dark.

Text is converted to outlines (Inter) so the SVGs render the same everywhere.
"""
import os

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

INTER = os.environ.get("INTER_FONT",
                       "/usr/share/fonts/opentype/inter/Inter-{}.otf")

THEMES = {
    "light": {"blue": "#2F5E9E", "orange": "#D9822B", "fill": "#F4DFC6",
              "text": "#17212C", "sub": "#5B6573"},
    "dark": {"blue": "#8DB2E8", "orange": "#E99A4C", "fill": "#3A2C20",
             "text": "#F3EFE6", "sub": "#B9C6D3"},
}


def star(cx, cy, r):
    return (f"M{cx},{cy - r} Q{cx},{cy} {cx + r},{cy} Q{cx},{cy} {cx},{cy + r} "
            f"Q{cx},{cy} {cx - r},{cy} Q{cx},{cy} {cx},{cy - r} Z")


def mark(t):
    """256x256 mark: jagged rows of array cells passing through a database."""
    cx, rx, ry, top, bot, sw = 176, 48, 14, 62, 196, 8
    left, right = cx - rx, cx + rx
    body = (f"M{left},{top} V{bot} A{rx},{ry} 0 0 0 {right},{bot} V{top} "
            f"A{rx},{ry} 0 0 0 {left},{top} Z")
    out = [f'<path d="{body}" fill="{t["fill"]}"/>']
    # rows: (y, number of cells outside the cylinder); 2 cells always inside
    rows = [(88, 2), (122, 0), (156, 4)]
    w, period, inside_x = 22, 28, (142, 170)
    for y, n_out in rows:
        # the first outside cell straddles the wall: the row passes through it
        xs = list(inside_x) + [114 - period * i for i in range(n_out)]
        for x in xs:
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{w}" rx="5" '
                       f'fill="{t["blue"]}"/>')
    # cylinder outline on top of the cells, then the top ellipse
    out.append(f'<path d="M{left},{top} V{bot} A{rx},{ry} 0 0 0 {right},{bot} '
               f'V{top}" fill="none" stroke="{t["orange"]}" stroke-width="{sw}" '
               f'stroke-linejoin="round"/>')
    out.append(f'<ellipse cx="{cx}" cy="{top}" rx="{rx}" ry="{ry}" '
               f'fill="{t["fill"]}" stroke="{t["orange"]}" stroke-width="{sw}"/>')
    # the agent: two sparks
    out.append(f'<path d="{star(54, 50, 30)}" fill="{t["orange"]}"/>')
    out.append(f'<path d="{star(92, 24, 12)}" fill="{t["blue"]}"/>')
    return "\n".join(out)


class Text:
    def __init__(self, weight):
        self.font = TTFont(INTER.format(weight))
        self.gs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        self.hmtx = self.font["hmtx"]

    def width(self, s, size, tracking=0.0):
        k = size / self.upm
        return sum(self.hmtx[self.cmap[ord(c)]][0] * k + tracking for c in s) - tracking

    def paths(self, s, x, baseline, size, fill, tracking=0.0):
        k = size / self.upm
        out = []
        for c in s:
            name = self.cmap[ord(c)]
            pen = SVGPathPen(self.gs)
            self.gs[name].draw(pen)
            d = pen.getCommands()
            if d:
                out.append(f'<path transform="translate({x:.2f},{baseline:.2f}) '
                           f'scale({k:.5f},{-k:.5f})" d="{d}" fill="{fill}"/>')
            x += self.hmtx[name][0] * k + tracking
        return "\n".join(out), x


def svg(width, height, body, bg=None):
    rect = f'<rect width="{width}" height="{height}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" '
            f'aria-label="awkward-monetizer">\n{rect}\n{body}\n</svg>\n')


def lockup(t):
    bold, med = Text("SemiBold"), Text("Medium")
    size, sub_size, tracking = 96, 26, 3.2
    mark_scale = 0.86                      # 256 * 0.86 = 220 px tall
    text_x = 256 * mark_scale + 36
    a, x = bold.paths("awkward", text_x, 128, size, t["blue"])
    b, x = bold.paths("-monetizer", x, 128, size, t["text"])
    c, x2 = med.paths("AGENTIC ANALYSIS ON RAGGED DATA", text_x + 4, 176,
                      sub_size, t["sub"], tracking)
    width = int(max(x, x2) + 24)
    body = (f'<g transform="translate(0,0) scale({mark_scale})">{mark(t)}</g>\n'
            f"{a}\n{b}\n{c}")
    return width, 220, body


if __name__ == "__main__":
    for name, t in THEMES.items():
        suffix = "" if name == "light" else "-dark"
        with open(f"awkward-monetizer-mark{suffix}.svg", "w") as fh:
            fh.write(svg(256, 256, mark(t)))
        w, h, body = lockup(t)
        with open(f"awkward-monetizer-logo{suffix}.svg", "w") as fh:
            fh.write(svg(w, h, body))
    print("written")
