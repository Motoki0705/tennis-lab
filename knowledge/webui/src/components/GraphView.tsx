"use client";
import { useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MarkerType,
  MiniMap,
} from "reactflow";
import "reactflow/dist/style.css";
import { layoutGraph } from "@/lib/layout";
import type { KnowledgeGraph } from "@/lib/types";
import { nodeTypes } from "./nodeTypes";

export function GraphView({
  graph,
  onSelect,
}: {
  graph: KnowledgeGraph;
  onSelect: (id: string) => void;
}) {
  const [relations, setRelations] = useState(false);
  const { nodes, edges } = useMemo(() => {
    const ids = new Set(graph.nodes.map((n) => n.id));
    const visibleEdges = graph.edges.filter(
      (e) =>
        ids.has(e.source) &&
        ids.has(e.target) &&
        (e.kind === "parent" || (relations && e.kind === "relation")),
    );
    return {
      nodes: layoutGraph(
        graph.nodes,
        visibleEdges.filter((e) => e.kind === "parent"),
      ),
      edges: visibleEdges.map((e) => ({
        ...e,
        label: e.label,
        style: {
          stroke: e.kind === "relation" ? "#8b83cb" : "#7f98a6",
          strokeDasharray: e.kind === "relation" ? "5 4" : undefined,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#7f98a6" },
      })),
    };
  }, [graph, relations]);
  return (
    <div className="graph-container">
      <label className="graph-toggle">
        <input
          type="checkbox"
          checked={relations}
          onChange={(e) => setRelations(e.target.checked)}
        />{" "}
        比較・確認などの関連線を表示
      </label>
      <ReactFlow
        key={`${graph.nodes.map((n) => n.id).join(",")}`}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, n) => onSelect(n.id)}
        fitView
        minZoom={0.04}
        nodesDraggable={false}
        nodesConnectable={false}
        onlyRenderVisibleElements
      >
        <Background color="#293846" gap={24} />
        <Controls />
        <MiniMap pannable zoomable nodeColor="#508f85" />
      </ReactFlow>
    </div>
  );
}
