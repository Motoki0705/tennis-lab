// Insert quoted labels and validated cell values from table_renderer.py.
#let ink = rgb("#17212d")
#let rule = rgb("#b1bece")
#let band = rgb("#edf3f9")
#set text(font: ("Noto Sans CJK JP", "DejaVu Sans"), lang: "ja", size: 9pt, fill: ink)
#set par(leading: 3pt)
#let numeric(value) = text(font: "DejaVu Sans Mono", value)
#let table-heading(label, unit) = {
  strong(label)
  if unit != "" {
    linebreak()
    text(size: 7.5pt, fill: rgb("#425369"), unit)
  }
}
#let dense-table(widths, alignment, title, notes, headings, cells) = {
  set page(width: widths.sum() + 16pt, height: auto, margin: 8pt, fill: white)
  text(size: 13pt, weight: "bold", title)
  linebreak()
  text(fill: rgb("#425369"), size: 8pt, notes)
  v(4pt)
  table(
    columns: widths,
    align: alignment,
    stroke: none,
    inset: (x: 4pt, y: 2pt),
    table.hline(stroke: (paint: rule, thickness: 0.85pt)),
    table.header(..headings),
    table.hline(stroke: (paint: rule, thickness: 0.5pt)),
    ..cells,
    table.hline(stroke: (paint: rule, thickness: 0.85pt)),
  )
}
