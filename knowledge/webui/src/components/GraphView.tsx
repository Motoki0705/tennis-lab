"use client";

import { useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  type Edge,
} from "reactflow";
import "reactflow/dist/style.css";

import { layoutGraph } from "@/lib/layout";
import type { KnowledgeGraph, KnowledgeNode } from "@/lib/types";
import { DetailPanel } from "./DetailPanel";
import { nodeTypes, PROVIDER_COLOR } from "./nodeTypes";

const EDGE_COLOR = "#8a8f98";

function issueList(issue?: number | number[]): number[] {
  if (issue == null) return [];
  return Array.isArray(issue) ? issue : [issue];
}

function uniqSorted<T>(xs: T[]): T[] {
  return Array.from(new Set(xs)).sort();
}

export function GraphView({ graph }: { graph: KnowledgeGraph }) {
  const allIssues = useMemo(
    () => uniqSorted(graph.nodes.flatMap((n) => issueList(n.issue))),
    [graph],
  );
  const allProviders = useMemo(
    () => uniqSorted(graph.nodes.map((n) => n.provider).filter((p): p is string => !!p)),
    [graph],
  );
  const allTags = useMemo(
    () => uniqSorted(graph.nodes.flatMap((n) => n.tags)),
    [graph],
  );

  const [issues, setIssues] = useState<Set<number>>(new Set());
  const [providers, setProviders] = useState<Set<string>>(new Set());
  const [tags, setTags] = useState<Set<string>>(new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const visible = useMemo(() => {
    return graph.nodes.filter((n) => {
      if (issues.size && !issueList(n.issue).some((i) => issues.has(i))) return false;
      if (providers.size && !(n.provider && providers.has(n.provider))) return false;
      if (tags.size && !n.tags.some((t) => tags.has(t))) return false;
      return true;
    });
  }, [graph, issues, providers, tags]);

  const { rfNodes, rfEdges } = useMemo(() => {
    const visIds = new Set(visible.map((n) => n.id));
    // Only parent→child dependency edges are drawn; group membership is shown
    // by enclosure (see layoutGraph), so member/relation edges are omitted.
    const parentEdges = graph.edges.filter(
      (e) => e.kind === "parent" && visIds.has(e.source) && visIds.has(e.target),
    );
    const rfNodes = layoutGraph(
      visible,
      parentEdges.map((e) => ({ source: e.source, target: e.target })),
    );
    const rfEdges: Edge[] = parentEdges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      style: { stroke: EDGE_COLOR, strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLOR, width: 16, height: 16 },
    }));
    return { rfNodes, rfEdges };
  }, [graph, visible]);

  const selected = useMemo(
    () => graph.nodes.find((n) => n.id === selectedId) ?? null,
    [graph, selectedId],
  );

  const toggle = <T,>(set: Set<T>, value: T, update: (s: Set<T>) => void) => {
    const next = new Set(set);
    next.has(value) ? next.delete(value) : next.add(value);
    update(next);
  };

  return (
    <div className="layout">
      <div className="filters">
        <div className="filters__group">
          <span className="filters__label">issue</span>
          {allIssues.map((i) => (
            <button
              key={i}
              className={`chip ${issues.has(i) ? "chip--on" : ""}`}
              onClick={() => toggle(issues, i, setIssues)}
            >
              #{i}
            </button>
          ))}
        </div>
        <div className="filters__group">
          <span className="filters__label">provider</span>
          {allProviders.map((p) => (
            <button
              key={p}
              className={`chip ${providers.has(p) ? "chip--on" : ""}`}
              style={providers.has(p) ? { background: PROVIDER_COLOR[p] ?? "#555", borderColor: "transparent" } : undefined}
              onClick={() => toggle(providers, p, setProviders)}
            >
              {p}
            </button>
          ))}
        </div>
        <details className="filters__tags">
          <summary className={`chip filters__tags-toggle ${tags.size ? "chip--on" : ""}`}>
            tags
            <span className="filters__tags-count">
              {tags.size ? `${tags.size} selected` : allTags.length}
            </span>
          </summary>
          <div className="filters__group filters__tags-options">
            {allTags.map((t) => (
              <button
                key={t}
                className={`chip ${tags.has(t) ? "chip--on" : ""}`}
                onClick={() => toggle(tags, t, setTags)}
              >
                #{t}
              </button>
            ))}
          </div>
        </details>
        {(issues.size || providers.size || tags.size) > 0 && (
          <button
            className="chip chip--clear"
            onClick={() => {
              setIssues(new Set());
              setProviders(new Set());
              setTags(new Set());
            }}
          >
            clear
          </button>
        )}
      </div>

      <div className="canvas">
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.2}
          proOptions={{ hideAttribution: true }}
          onNodeClick={(_, n) => setSelectedId(n.id)}
          onPaneClick={() => setSelectedId(null)}
        >
          <Background color="#2a2f36" gap={20} />
          <Controls />
          <MiniMap
            pannable
            zoomable
            nodeColor={(n) =>
              n.type === "groupNode"
                ? "#f5d76e"
                : PROVIDER_COLOR[(n.data as KnowledgeNode).provider ?? "other"] ?? "#777"
            }
            maskColor="rgba(10,12,15,0.7)"
          />
        </ReactFlow>
        <DetailPanel node={selected} onClose={() => setSelectedId(null)} />
      </div>
    </div>
  );
}
