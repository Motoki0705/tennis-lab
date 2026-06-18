import dagre from "dagre";
import type { Edge, Node } from "reactflow";

export const NODE_W = 260;
export const RUN_H = 150;
export const GROUP_H = 96;

export function layout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 90, marginx: 40, marginy: 40 });

  for (const n of nodes) {
    const h = n.type === "groupNode" ? GROUP_H : RUN_H;
    g.setNode(n.id, { width: NODE_W, height: h });
  }
  for (const e of edges) {
    g.setEdge(e.source, e.target);
  }
  dagre.layout(g);

  return nodes.map((n) => {
    const pos = g.node(n.id);
    const h = n.type === "groupNode" ? GROUP_H : RUN_H;
    return {
      ...n,
      position: { x: pos.x - NODE_W / 2, y: pos.y - h / 2 },
    };
  });
}
