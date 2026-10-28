# Logo

| File | Use |
|---|---|
| `awkward-monetizer-logo.svg` | mark + wordmark, light backgrounds |
| `awkward-monetizer-logo-dark.svg` | mark + wordmark, dark backgrounds |
| `awkward-monetizer-mark.svg` | mark only (favicon, avatar, slide corner), light |
| `awkward-monetizer-mark-dark.svg` | mark only, dark |

The mark shows jagged rows of array cells passing through a database: Awkward
Array (blue) and MonetDB (orange), with sparks for the agent. Text is converted
to outlines (Inter), so the SVGs render the same everywhere.

`make_logo.py` regenerates the SVGs (needs `fonttools` and the Inter font; point
`INTER_FONT` at it, e.g. `INTER_FONT=~/Library/Fonts/Inter-{}.otf`).

PNG exports are not tracked (the repository ignores `*.png`); render them
from the SVGs when needed, e.g. for a GitHub social preview.
