import { promises as fs } from "node:fs";
import path from "node:path";
import { RUNS_DIR } from "@/lib/content";
export const dynamic = "force-dynamic";
const MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
};
export async function GET(
  _request: Request,
  { params }: { params: { path: string[] } },
) {
  if (
    params.path.some(
      (part) =>
        !/^[a-zA-Z0-9_.-]+$/.test(part) || part === "." || part === "..",
    )
  )
    return new Response("bad path", { status: 400 });
  const file = path.join(RUNS_DIR, ...params.path);
  const mime = MIME[path.extname(file).toLowerCase()];
  if (!mime) return new Response("unsupported asset", { status: 400 });
  try {
    const real = await fs.realpath(file);
    if (!real.startsWith((await fs.realpath(RUNS_DIR)) + path.sep))
      return new Response("not found", { status: 404 });
    return new Response(await fs.readFile(real), {
      headers: { "Content-Type": mime, "X-Content-Type-Options": "nosniff" },
    });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT")
      return new Response("not found", { status: 404 });
    throw error;
  }
}
