import { ResearchExplorer } from "@/components/ResearchExplorer";
import { getGraph } from "@/lib/nodes";
export const dynamic = "force-dynamic";
export default async function Page() { return <ResearchExplorer graph={await getGraph()} />; }
