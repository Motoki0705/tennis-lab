"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import dagre from "dagre";
import ReactFlow, {
  Background,
  Controls,
  MarkerType,
  Position,
  type ReactFlowInstance,
} from "reactflow";
import "reactflow/dist/style.css";
import { ROLE_LABELS, TEXT_LABELS } from "@/lib/citation-types";
import type { CitationPaper } from "@/lib/citation-types";
import type { CitationRelation } from "@/lib/citations";

const SHORT_ROLES = {
  foundation: "基盤",
  component: "個別技術",
  comparison: "比較対象",
  related: "関連研究",
  benchmark: "データセット",
};

export function CitationMap({
  root,
  relations,
  selected,
  onSelect,
}: {
  root: CitationPaper;
  relations: CitationRelation[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [instance, setInstance] = useState<ReactFlowInstance | null>(null);
  useEffect(() => {
    if (!instance || !container.current) return;
    let frame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        void instance.fitView({ padding: 0.18 });
      });
    });
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [instance]);
  const { nodes, edges } = useMemo(() => {
    const papers = new Map([[root.id, root]]);
    for (const rel of relations) {
      papers.set(rel.source.id, rel.source);
      if (rel.target) papers.set(rel.target.id, rel.target);
    }
    const unresolved = relations.filter((rel) => !rel.target);
    const ids = [
      ...papers.keys(),
      ...unresolved.map((rel) => `unresolved:${rel.id}`),
    ];
    const layout = new dagre.graphlib.Graph();
    layout.setGraph({
      rankdir: "LR",
      nodesep: 50,
      ranksep: 210,
      marginx: 15,
      marginy: 15,
    });
    layout.setDefaultEdgeLabel(() => ({}));
    for (const id of ids) layout.setNode(id, { width: 236, height: 116 });
    const edges = relations.map((rel) => {
      const target = rel.target?.id ?? `unresolved:${rel.id}`;
      layout.setEdge(rel.source.id, target);
      return {
        id: rel.id,
        source: rel.source.id,
        target,
        label: rel.roles.length
          ? rel.roles.map((r) => SHORT_ROLES[r]).join(" / ")
          : "役割未確定",
        style: { stroke: "#8ebbae", strokeWidth: 1.5 },
        labelStyle: { fill: "#d5eee5", fontSize: 12 },
        labelBgStyle: { fill: "#152a30" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#8ebbae" },
        ariaLabel: `${rel.source.title} → ${rel.target?.title ?? "未同定"}: ${rel.roles.map((r) => ROLE_LABELS[r]).join(" / ")}`,
      };
    });
    dagre.layout(layout);
    const nodes = ids.map((id) => {
      const paper = papers.get(id);
      const rel = unresolved.find((r) => `unresolved:${r.id}` === id);
      const position = layout.node(id);
      const selection = paper?.id ?? rel!.source.id;
      return {
        id,
        position: { x: position.x - 118, y: position.y - 58 },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        className: `citation-flow-node ${paper?.id === selected ? "selected" : ""}`,
        style: { width: 236, padding: 0, pointerEvents: "all" as const },
        data: {
          label: (
            <button
              className="citation-node"
              onClick={() => onSelect(selection)}
              aria-label={`${paper?.title ?? "未同定の参考文献"} の引用詳細`}
            >
              <small>
                {id === root.id ? "起点 · " : ""}
                {paper ? TEXT_LABELS[paper.text_status] : "未同定"}
              </small>
              <strong>{paper?.title ?? rel!.references[0].citation}</strong>
              <span>
                {paper
                  ? `${paper.year ?? "年不明"} · ${paper.external_ids.doi ?? paper.external_ids.arxiv ?? paper.id}`
                  : rel!.references[0].ref_id}
              </span>
            </button>
          ),
        },
      };
    });
    return { nodes, edges };
  }, [root, relations, selected, onSelect]);
  return (
    <div ref={container} className="citation-canvas" aria-label="引用マップ">
      <ReactFlow
        key={nodes.map((n) => n.id).join(",")}
        nodes={nodes}
        edges={edges}
        onInit={setInstance}
        fitView
        minZoom={0.08}
        nodesDraggable={false}
        nodesConnectable={false}
        nodesFocusable={false}
        edgesFocusable={false}
        elementsSelectable={false}
      >
        <Background color="#30454c" gap={24} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
