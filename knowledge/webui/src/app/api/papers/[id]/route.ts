import { promises as fs } from "node:fs";
import path from "node:path";
import { PAPERS_DIR } from "@/lib/content";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: { id: string } },
) {
  if (!/^paper-[0-9]{4}-[a-z0-9]+(?:-[a-z0-9]+)*$/.test(params.id))
    return new Response("bad id", { status: 400 });
  const file = path.join(PAPERS_DIR, params.id, "paper.pdf");
  try {
    const real = await fs.realpath(file);
    if (!real.startsWith((await fs.realpath(PAPERS_DIR)) + path.sep))
      return new Response("not found", { status: 404 });
    const data = await fs.readFile(real);
    return new Response(data, {
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `inline; filename="${params.id}.pdf"`,
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT")
      return new Response("not found", { status: 404 });
    throw error;
  }
}
