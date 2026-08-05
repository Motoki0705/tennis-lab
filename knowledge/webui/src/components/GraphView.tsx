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
import type { KnowledgeGraph, KnowledgeNode, NodeType } from "@/lib/types";
import { DetailPanel } from "./DetailPanel";
import { nodeAccent, nodeTypes } from "./nodeTypes";

const EDGE_COLOR = "#8a8f98";

function issueList(issue?: number | number[]): number[] {
  if (issue == null) return [];
  return Array.isArray(issue) ? issue : [issue];
}

function uniqSorted<T>(items: T[]): T[] {
  return Array.from(new Set(items)).sort();
}

export function GraphView({ graph }: { graph: KnowledgeGraph }) {
  const allIssues = useMemo(
    () => uniqSorted(graph.nodes.flatMap((node) => issueList(node.issue))),
    [graph],
  );
  const allProviders = useMemo(
    () =>
      uniqSorted(
        graph.nodes
          .map((node) => node.provider)
          .filter((provider): provider is string => !!provider),
      ),
    [graph],
  );
  const allTypes = useMemo(
    () => uniqSorted(graph.nodes.map((node) => node.type)),
    [graph],
  );
  const allTags = useMemo(
    () => uniqSorted(graph.nodes.flatMap((node) => node.tags)),
    [graph],
  );

  const [issues, setIssues] = useState<Set<number>>(new Set());
  const [providers, setProviders] = useState<Set<string>>(new Set());
  const [selectedTypes, setSelectedTypes] = useState<Set<NodeType>>(new Set());
  const [tags, setTags] = useState<Set<string>>(new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const visible = useMemo(() => {
    return graph.nodes.filter((node) => {
      if (issues.size && !issueList(node.issue).some((issue) => issues.has(issue))) {
        return false;
      }
      if (providers.size && !(node.provider && providers.has(node.provider))) {
        return false;
      }
      if (selectedTypes.size && !selectedTypes.has(node.type)) return false;
      if (tags.size && !node.tags.some((tag) => tags.has(tag))) return false;
      return true;
    });
  }, [graph, issues, providers, selectedTypes, tags]);

  const { rfNodes, rfEdges } = useMemo(() => {
    const visibleIds = new Set(visible.map((node) => node.id));
    const parentEdges = graph.edges.filter(
      (edge) =>
        edge.kind === "parent" &&
        visibleIds.has(edge.source) &&
        visibleIds.has(edge.target),
    );
    const rfNodes = layoutGraph(
      visible,
      parentEdges.map((edge) => ({ source: edge.source, target: edge.target })),
    );
    const rfEdges: Edge[] = parentEdges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      style: { stroke: EDGE_COLOR, strokeWidth: 1.5 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: EDGE_COLOR,
        width: 16,
        height: 16,
      },
    }));
    return { rfNodes, rfEdges };
  }, [graph, visible]);

  const selected = useMemo(
    () => graph.nodes.find((node) => node.id === selectedId) ?? null,
    [graph, selectedId],
  );

  const toggle = <T,>(set: Set<T>, value: T, update: (next: Set<T>) => void) => {
    const next = new Set(set);
    next.has(value) ? next.delete(value) : next.add(value);
    update(next);
  };

  const hasFilters =
    issues.size > 0 || providers.size > 0 || selectedTypes.size > 0 || tags.size > 0;

  return (
    <div className="layout">
      <div className="filters">
        <div className="filters__group">
          <span className="filters__label">type</span>
          {allTypes.map((type) => (
            <button
              key={type}
              className={`chip ${selectedTypes.has(type) ? "chip--on" : ""}`}
              onClick={() => toggle(selectedTypes, type, setSelectedTypes)}
            >
              {type}
            </button>
          ))}
        </div>
        <div className="filters__group">
          <span className="filters__label">issue</span>
          {allIssues.map((issue) => (
            <button
              key={issue}
              className={`chip ${issues.has(issue) ? "chip--on" : ""}`}
              onClick={() => toggle(issues, issue, setIssues)}
            >
              #{issue}
            </button>
          ))}
        </div>
        <div className="filters__group">
          <span className="filters__label">provider</span>
          {allProviders.map((provider) => (
            <button
              key={provider}
              className={`chip ${providers.has(provider) ? "chip--on" : ""}`}
              onClick={() => toggle(providers, provider, setProviders)}
            >
              {provider}
            </button>
          ))}
        </div>
        <div className="filters__group">
          <span className="filters__label">tag</span>
          {allTags.map((tag) => (
            <button
              key={tag}
              className={`chip ${tags.has(tag) ? "chip--on" : ""}`}
              onClick={() => toggle(tags, tag, setTags)}
            >
              #{tag}
            </button>
          ))}
        </div>
        {hasFilters && (
          <button
            className="chip chip--clear"
            onClick={() => {
              setIssues(new Set());
              setProviders(new Set());
              setSelectedTypes(new Set());
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
          onNodeClick={(_, node) => setSelectedId(node.id)}
          onPaneClick={() => setSelectedId(null)}
        >
          <Background color="#2a2f36" gap={20} />
          <Controls />
          <MiniMap
            pannable
            zoomable
            nodeColor={(node) =>
              node.type === "groupNode"
                ? "#f5d76e"
                : nodeAccent(node.data as KnowledgeNode)
            }
            maskColor="rgba(10,12,15,0.7)"
          />
        </ReactFlow>
        <DetailPanel node={selected} onClose={() => setSelectedId(null)} />
      </div>
    </div>
  );
}
