import { promises as fs } from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";

import { RUNS_DIR } from "@/lib/content";

// Node ids are lowercase [a-z0-9-]; reject anything else to avoid path traversal.
const ID_RE = /^[a-z0-9-]+$/;

export async function GET(
  _req: Request,
  { params }: { params: { id: string } },
) {
  const { id } = params;
  if (!ID_RE.test(id)) {
    return new Response("bad id", { status: 400 });
  }
  const file = path.join(RUNS_DIR, id, "curves.png");
  try {
    const buf = await fs.readFile(file);
    return new Response(buf, {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "no-cache",
      },
    });
  } catch {
    return new Response("not found", { status: 404 });
  }
}
