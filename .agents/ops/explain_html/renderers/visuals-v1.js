/* Immutable managed renderer v1. No eval, network, imports or user code. */
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const colors = ["#005a9c", "#a33500", "#367328", "#7a3589"];
  const number = x => {
    if (typeof x !== "number" || !Number.isFinite(x)) throw new Error("Invalid numeric graph input");
    return x;
  };
  const fmt = x => String(Number(number(x).toPrecision(9)));

  // Inputs have passed visual_definitions.py. This runtime also checks numeric
  // boundaries before touching the DOM; it interprets no expressions or URLs.
  function series(definition, value) {
    if (definition.type === "comparison") {
      return definition.series.map(s => ({label: s.label, points: s.points.map(p => ({x: number(p.x), y: number(p.y)}))}));
    }
    if (definition.type !== "observation_error") throw new Error("Unknown managed graph type");
    const slider = definition.slider, data = definition.data;
    const error = value === undefined ? number(slider.initial) : number(value);
    if (error < slider.min || error > slider.max || data.disparity_px + slider.min <= 0) throw new Error("Observation error is outside the declared range");
    const inputs = [...new Set([...data.sample_errors_px, error])].sort((a, b) => a - b);
    return [{label: definition.y.label, points: inputs.map(x => ({
      x: number(x),
      y: number(number(data.focal_px) * number(data.baseline_m) / (number(data.disparity_px) + number(x)))
    }))}];
  }

  function svgNode(tag, attributes, text) {
    const node = document.createElementNS(NS, tag);
    for (const [name, value] of Object.entries(attributes || {})) node.setAttribute(name, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function plot(definition, rows) {
    const [x0, x1] = definition.x.domain, [y0, y1] = definition.y.domain;
    if (!(x1 > x0 && y1 > y0)) throw new Error("Invalid graph domain");
    const px = x => 72 + (x - x0) / (x1 - x0) * 544;
    const py = y => 262 - (y - y0) / (y1 - y0) * 242;
    const svg = svgNode("svg", {xmlns: NS, viewBox: "0 0 640 320", role: "img", width: "100%", "aria-label": `${definition.y.label} (${definition.y.unit}) against ${definition.x.label} (${definition.x.unit})`});
    svg.append(svgNode("path", {d: "M72 20V262H616", fill: "none", stroke: "currentColor"}));
    rows.forEach((row, index) => {
      const color = colors[index % colors.length];
      svg.append(svgNode("polyline", {points: row.points.map(p => `${fmt(px(p.x))},${fmt(py(p.y))}`).join(" "), fill: "none", stroke: color, "stroke-width": "2"}));
      row.points.forEach(p => {
        const point = svgNode("circle", {cx: fmt(px(p.x)), cy: fmt(py(p.y)), r: "4", fill: color, "data-x": fmt(p.x), "data-y": fmt(p.y)});
        point.append(svgNode("title", {}, `${row.label}: ${fmt(p.x)}, ${fmt(p.y)}`));
        svg.append(point);
      });
    });
    svg.append(
      svgNode("text", {x: "330", y: "310", "text-anchor": "middle"}, `${definition.x.label} (${definition.x.unit})`),
      svgNode("text", {x: "8", y: "14"}, `${definition.y.label} (${definition.y.unit})`),
      svgNode("text", {x: "72", y: "282"}, fmt(x0)),
      svgNode("text", {x: "586", y: "282"}, fmt(x1)),
      svgNode("text", {x: "28", y: "262"}, fmt(y0)),
      svgNode("text", {x: "28", y: "30"}, fmt(y1))
    );
    return svg;
  }

  function render(container, definition, value) {
    const rows = series(definition, value);
    container.querySelector("[data-plot]").replaceChildren(plot(definition, rows));
    const table = container.querySelector("[data-values]");
    table.replaceChildren();
    for (const row of rows) {
      for (const p of row.points) {
        const tr = document.createElement("tr");
        for (const [index, text] of [row.label, fmt(p.x), fmt(p.y)].entries()) {
          const cell = document.createElement(index === 0 ? "th" : "td");
          if (index === 0) cell.scope = "row";
          cell.textContent = text;
          tr.append(cell);
        }
        table.append(tr);
      }
    }
    if (definition.type === "observation_error") {
      const current = value === undefined ? definition.slider.initial : value;
      const depth = rows[0].points.find(p => p.x === current).y;
      container.querySelector("[data-slider-value]").textContent = `${fmt(current)} px; ${definition.y.label} = ${fmt(depth)} m`;
    }
  }

  function mount(scope) {
    for (const container of scope.querySelectorAll('[data-managed-graph="1"]')) {
      const definition = JSON.parse(container.getAttribute("data-definition"));
      render(container, definition);
      const slider = container.querySelector("[data-managed-slider]");
      if (slider) slider.addEventListener("input", () => render(container, definition, Number(slider.value)));
    }
  }
  const api = Object.freeze({version: "1", series, mount});
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ExplainVisualsV1 = api;
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => mount(document), {once: true});
    else mount(document);
  }
})(typeof globalThis === "undefined" ? this : globalThis);
