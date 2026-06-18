import { GraphView } from "@/components/GraphView";
import { getGraph } from "@/lib/nodes";

export const dynamic = "force-dynamic";

export default async function Page() {
  const graph = await getGraph();
  const runs = graph.nodes.filter((n) => n.type === "run").length;
  const groups = graph.nodes.filter((n) => n.type === "group").length;

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>Knowledge Control</h1>
          <span className="subtitle">tennis-lab 学習知識グラフ</span>
        </div>
        <div className="stats">
          <span>{runs} runs</span>
          <span>{groups} groups</span>
          <span className="legend">
            <i className="ln ln--parent" /> parent→child
            <i className="ln ln--member" /> group
            <i className="ln ln--relation" /> relation
          </span>
        </div>
      </header>
      {graph.nodes.length === 0 ? (
        <p className="empty">
          knowledge/nodes に .md ノードがありません。SKILL の手順でノードを追加してください。
        </p>
      ) : (
        <GraphView graph={graph} />
      )}
    </main>
  );
}
