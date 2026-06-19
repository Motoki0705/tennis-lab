import dagre from "dagre";
import type { Node } from "reactflow";

import type { KnowledgeNode } from "./types";

export const NODE_W = 260;
export const RUN_H = 150;
// Space reserved at the top of a group box for its header label.
export const GROUP_HEADER = 48;
// Inner padding between a group box edge and the member nodes it encloses.
export const GROUP_PAD = 20;

type SimpleEdge = { source: string; target: string };
type Size = { w: number; h: number };
type Pos = { x: number; y: number };

/**
 * Run a dagre top-to-bottom layout over the given ids and return positions
 * (top-left, normalized so the content starts at the origin) plus the overall
 * content size.
 */
function dagreLayout(
  ids: string[],
  sizes: Map<string, Size>,
  edges: SimpleEdge[],
): { pos: Map<string, Pos>; width: number; height: number } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 50, ranksep: 80, marginx: 8, marginy: 8 });

  for (const id of ids) {
    const s = sizes.get(id)!;
    g.setNode(id, { width: s.w, height: s.h });
  }
  for (const e of edges) {
    if (sizes.has(e.source) && sizes.has(e.target)) g.setEdge(e.source, e.target);
  }
  dagre.layout(g);

  const raw = new Map<string, Pos>();
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const id of ids) {
    const n = g.node(id);
    const s = sizes.get(id)!;
    const x = n.x - s.w / 2;
    const y = n.y - s.h / 2;
    raw.set(id, { x, y });
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + s.w);
    maxY = Math.max(maxY, y + s.h);
  }
  if (!Number.isFinite(minX)) {
    return { pos: new Map(), width: 0, height: 0 };
  }
  const pos = new Map<string, Pos>();
  for (const [id, p] of raw) pos.set(id, { x: p.x - minX, y: p.y - minY });
  return { pos, width: maxX - minX, height: maxY - minY };
}

/**
 * Lay out the knowledge graph with groups as enclosing container nodes: each
 * group's member runs are positioned *inside* the group box (React Flow
 * parent/child), and only parent→child dependency edges are drawn. Group
 * membership is expressed by containment, not by edges.
 */
export function layoutGraph(
  nodes: KnowledgeNode[],
  parentEdges: SimpleEdge[],
): Node<KnowledgeNode>[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const groups = nodes.filter((n) => n.type === "group");
  const runs = nodes.filter((n) => n.type === "run");

  // run id -> the (visible) group that first claims it as a member.
  const groupOf = new Map<string, string>();
  for (const g of groups) {
    for (const m of g.members) {
      if (byId.has(m) && !groupOf.has(m)) groupOf.set(m, g.id);
    }
  }

  // 1. Lay out each group's members internally and size its box.
  const groupBox = new Map<string, Size>();
  const childRel = new Map<string, Pos>();
  for (const g of groups) {
    const memberIds = g.members.filter(
      (m) => byId.has(m) && groupOf.get(m) === g.id,
    );
    if (memberIds.length === 0) {
      groupBox.set(g.id, {
        w: NODE_W + GROUP_PAD * 2,
        h: GROUP_HEADER + GROUP_PAD,
      });
      continue;
    }
    const sizes = new Map(memberIds.map((m) => [m, { w: NODE_W, h: RUN_H }]));
    const memberSet = new Set(memberIds);
    const intra = parentEdges.filter(
      (e) => memberSet.has(e.source) && memberSet.has(e.target),
    );
    const { pos, width, height } = dagreLayout(memberIds, sizes, intra);
    for (const m of memberIds) {
      const p = pos.get(m)!;
      childRel.set(m, { x: p.x + GROUP_PAD, y: p.y + GROUP_HEADER });
    }
    groupBox.set(g.id, {
      w: width + GROUP_PAD * 2,
      h: GROUP_HEADER + height + GROUP_PAD,
    });
  }

  // 2. Top-level layout over groups + ungrouped runs. Parent edges are mapped
  //    to their containing top-level node; intra-group edges are dropped here.
  const freeRuns = runs.filter((r) => !groupOf.has(r.id));
  const topIds = [...groups.map((g) => g.id), ...freeRuns.map((r) => r.id)];
  const topSizes = new Map<string, Size>();
  for (const g of groups) topSizes.set(g.id, groupBox.get(g.id)!);
  for (const r of freeRuns) topSizes.set(r.id, { w: NODE_W, h: RUN_H });

  const top = (id: string) => groupOf.get(id) ?? id;
  const topEdges: SimpleEdge[] = [];
  const seen = new Set<string>();
  for (const e of parentEdges) {
    const s = top(e.source);
    const t = top(e.target);
    if (s === t) continue;
    const key = `${s}->${t}`;
    if (seen.has(key)) continue;
    seen.add(key);
    topEdges.push({ source: s, target: t });
  }
  const { pos: topPos } = dagreLayout(topIds, topSizes, topEdges);

  // 3. Assemble React Flow nodes. Parents (groups) must precede their children.
  const out: Node<KnowledgeNode>[] = [];
  for (const g of groups) {
    const p = topPos.get(g.id) ?? { x: 0, y: 0 };
    const box = groupBox.get(g.id)!;
    out.push({
      id: g.id,
      type: "groupNode",
      data: g,
      position: { x: p.x, y: p.y },
      style: { width: box.w, height: box.h },
    });
  }
  for (const r of runs) {
    const gid = groupOf.get(r.id);
    if (gid) {
      out.push({
        id: r.id,
        type: "runNode",
        data: r,
        parentNode: gid,
        extent: "parent",
        position: childRel.get(r.id) ?? { x: GROUP_PAD, y: GROUP_HEADER },
      });
    } else {
      const p = topPos.get(r.id) ?? { x: 0, y: 0 };
      out.push({ id: r.id, type: "runNode", data: r, position: { x: p.x, y: p.y } });
    }
  }
  return out;
}
