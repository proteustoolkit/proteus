# Proteus logo

Hand-written SVG (every file has a `viewBox`, no embedded rasters, no font references: the wordmark is outlined).
The trident was designed by cekees's daughter; the rest is built around it.

## primary/ (the logos to use)

Grey trident with no container; organization-agnostic.

- `proteus-mark.svg`: trident `#898b8e`; works on light and dark backgrounds (3.4:1 on white, 4-5:1 on dark).
- `proteus-mark-light-background.svg`: trident `#5f6164`, for white backgrounds where more contrast is wanted (6.2:1).
- `proteus-mark-auto.svg`: `#5f6164` in light mode and `#898b8e` in dark mode (`prefers-color-scheme`; works through `<img>`).
- `proteus-lockup-{h,v}-{light-background,dark-background,auto}.svg`: trident + PROTEUS, horizontal and stacked
  (stacked: the stem sits over the T). The suffix names the background the file is meant for.
- `proteus-favicon.svg`, `proteus-favicon.ico` (16/32/48): square, trident alone, transparent; heavier than the mark (3-unit round stroke, `#767a7e`, fills the square) so it stays legible at 16 px in a browser tab.
- `png/`: transparent PNG renders (mark 512/1024, lockups 1200 and 800 wide, favicon 16 to 512).

## lsu/ (secondary: LSU-affiliated uses)

Purple `#592c82` chamfered-corner badge with a gold `#ffc72a` edge and the grey trident: `proteus-badge.svg`,
a square `proteus-favicon.svg`, lockups, `.ico`, `png/`.

## Where each is used

| Use | File |
| --- | --- |
| Sphinx sidebar (`html_logo`, `docs/source/conf.py`) | `primary/proteus-lockup-v-light-background.svg` |
| Sphinx favicon (`html_favicon`) | `primary/proteus-favicon.ico` |
| Top-level `README.md` | `primary/proteus-lockup-h-{light,dark}-background.svg` via `<picture>` |
| Commented-out Doxygen and LaTeX logos | `primary/png/proteus-favicon-64.png`, `primary/png/proteus-lockup-v-light-background-800.png` |

## Colours and sources

- The grey, purple and gold follow the swatches in the LSU CCT logo files: LSU Logo Grey 50% / PANTONE Cool Gray 8 C,
  LSU Purple CMYK Coated 82/98/0/12 / PANTONE 268 C, LSU Gold CMYK Coated 0/19/89/0 / PANTONE 123 C.
  The sRGB values are LAB-to-sRGB conversions of those swatches, not values published by LSU; check them against the
  official LSU brand assets before print or formal use.
- `#5f6164` is a derived darker grey, not an LSU colour.
- Wordmark: Montserrat SemiBold (SIL Open Font License, see `Montserrat-OFL.txt`), outlined, letter-spaced, not kerned.
- The badge's polygonal outline echoes the angular LSU cap lettering without reproducing it. LSU marks belong to LSU:
  confirm the badge is acceptable to LSU before using it outside this project.
- No CMYK, EPS or PDF exports exist yet.

## Previous logo

The old Army Corps castle-style logo (`../corps_logo_cross2.svg`) and the old icons (`../triton*.ico`, `../proteus*.png`,
`../proteusSimple.ico`, `../proteus.pdf`) are kept for the applications that need them; nothing in the docs build
references them any more.
