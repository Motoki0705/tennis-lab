import dagre from "dagre";
import type { Node } from "reactflow";

import type { KnowledgeNode } from "./types";

export const NODE_W = 260;
export const ENTITY_H = 150;
export const GROUP_HEADER = 48;
export const GROUP_PAD = 20;

type SimpleEdge = { source: string; target: string };
type Size = { w: number; h: number };
type Pos = { x: number; y: number };

function dagreLayout(
  ids: string[],
  sizes: Map<string, Size>,
  edges: SimpleEdge[],
): { pos: Map<string, Pos>; width: number; height: number } {
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: "TB", nodesep: 50, ranksep: 80, marginx: 8, marginy: 8 });

  for (const id of ids) {
    const size = sizes.get(id)!;
    graph.setNode(id, { width: size.w, height: size.h });
  }
  for (const edge of edges) {
    if (sizes.has(edge.source) && sizes.has(edge.target)) {
      graph.setEdge(edge.source, edge.target);
    }
  }
  dagre.layout(graph);

  const raw = new Map<string, Pos>();
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const id of ids) {
    const node = graph.node(id);
    const size = sizes.get(id)!;
    const x = node.x - size.w / 2;
    const y = node.y - size.h / 2;
    raw.set(id, { x, y });
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + size.w);
    maxY = Math.max(maxY, y + size.h);
  }
  if (!Number.isFinite(minX)) {
    return { pos: new Map(), width: 0, height: 0 };
  }
  const pos = new Map<string, Pos>();
  for (const [id, point] of raw) {
    pos.set(id, { x: point.x - minX, y: point.y - minY });
  }
  return { pos, width: maxX - minX, height: maxY - minY };
}

/**
 * Lay out all formal node types. Groups remain enclosing containers for their
 * members; run, paper and proposal nodes are regular graph entities.
 */
export function layoutGraph(
  nodes: KnowledgeNode[],
  parentEdges: SimpleEdge[],
): Node<KnowledgeNode>[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const groups = nodes.filter((node) => node.type === "group");
  const entities = nodes.filter((node) => node.type !== "group");

  const groupOf = new Map<string, string>();
  for (const group of groups) {
    for (const member of group.members) {
      if (byId.has(member) && !groupOf.has(member)) groupOf.set(member, group.id);
    }
  }

  const groupBox = new Map<string, Size>();
  const childRel = new Map<string, Pos>();
  for (const group of groups) {
    const memberIds = group.members.filter(
      (member) => byId.has(member) && groupOf.get(member) === group.id,
    );
    if (memberIds.length === 0) {
      groupBox.set(group.id, {
        w: NODE_W + GROUP_PAD * 2,
        h: GROUP_HEADER + GROUP_PAD,
      });
      continue;
    }
    const sizes = new Map(
      memberIds.map((member) => [member, { w: NODE_W, h: ENTITY_H }]),
    );
    const memberSet = new Set(memberIds);
    const intra = parentEdges.filter(
      (edge) => memberSet.has(edge.source) && memberSet.has(edge.target),
    );
    const { pos, width, height } = dagreLayout(memberIds, sizes, intra);
    for (const member of memberIds) {
      const point = pos.get(member)!;
      childRel.set(member, {
        x: point.x + GROUP_PAD,
        y: point.y + GROUP_HEADER,
      });
    }
    groupBox.set(group.id, {
      w: width + GROUP_PAD * 2,
      h: GROUP_HEADER + height + GROUP_PAD,
    });
  }

  const freeEntities = entities.filter((entity) => !groupOf.has(entity.id));
  const topIds = [
    ...groups.map((group) => group.id),
    ...freeEntities.map((entity) => entity.id),
  ];
  const topSizes = new Map<string, Size>();
  for (const group of groups) topSizes.set(group.id, groupBox.get(group.id)!);
  for (const entity of freeEntities) {
    topSizes.set(entity.id, { w: NODE_W, h: ENTITY_H });
  }

  const top = (id: string) => groupOf.get(id) ?? id;
  const topEdges: SimpleEdge[] = [];
  const seen = new Set<string>();
  for (const edge of parentEdges) {
    const source = top(edge.source);
    const target = top(edge.target);
    if (source === target) continue;
    const key = `${source}->${target}`;
    if (seen.has(key)) continue;
    seen.add(key);
    topEdges.push({ source, target });
  }
  const { pos: topPos } = dagreLayout(topIds, topSizes, topEdges);

  const output: Node<KnowledgeNode>[] = [];
  for (const group of groups) {
    const point = topPos.get(group.id) ?? { x: 0, y: 0 };
    const box = groupBox.get(group.id)!;
    output.push({
      id: group.id,
      type: "groupNode",
      data: group,
      position: { x: point.x, y: point.y },
      style: { width: box.w, height: box.h },
    });
  }
  for (const entity of entities) {
    const groupId = groupOf.get(entity.id);
    if (groupId) {
      output.push({
        id: entity.id,
        type: "runNode",
        data: entity,
        parentNode: groupId,
        extent: "parent",
        position: childRel.get(entity.id) ?? { x: GROUP_PAD, y: GROUP_HEADER },
      });
    } else {
      const point = topPos.get(entity.id) ?? { x: 0, y: 0 };
      output.push({
        id: entity.id,
        type: "runNode",
        data: entity,
        position: { x: point.x, y: point.y },
      });
    }
  }
  return output;
}
