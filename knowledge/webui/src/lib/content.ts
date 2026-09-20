import { promises as fs } from "node:fs";
import path from "node:path";
import { marked } from "marked";
import sanitizeHtml from "sanitize-html";

export const KNOWLEDGE_DIR =
  process.env.KNOWLEDGE_DIR ?? path.resolve(process.cwd(), "..");
export const NODES_DIR =
  process.env.KNOWLEDGE_NODES_DIR ?? path.join(KNOWLEDGE_DIR, "nodes");
export const RUNS_DIR =
  process.env.KNOWLEDGE_RUNS_DIR ?? path.join(KNOWLEDGE_DIR, "runs");
export const PAPERS_DIR = path.join(KNOWLEDGE_DIR, "Papers");

export async function markdownFiles(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const file = path.join(dir, entry.name);
      return entry.isDirectory()
        ? markdownFiles(file)
        : entry.isFile() && entry.name.endsWith(".md")
          ? [file]
          : [];
    }),
  );
  return files.flat().sort();
}

export function renderMarkdown(
  content: string,
  file: string,
  targets: Map<string, string>,
): string {
  return sanitizeHtml(marked.parse(content, { async: false }) as string, {
    allowedTags: [...sanitizeHtml.defaults.allowedTags, "img"],
    allowedAttributes: {
      ...sanitizeHtml.defaults.allowedAttributes,
      img: ["src", "alt", "title"],
    },
    transformTags: {
      img: (_tag, attrs) => {
        const src = attrs.src ?? "";
        if (!/^(?:[a-z]+:|\/)/i.test(src)) {
          const resolved = path.resolve(path.dirname(file), src);
          const relative = path.relative(RUNS_DIR, resolved);
          if (!relative.startsWith("..") && !path.isAbsolute(relative))
            attrs.src = `/api/assets/${relative}`;
          else {
            const repoPath = path.relative(
              path.resolve(KNOWLEDGE_DIR, ".."),
              resolved,
            );
            if (!repoPath.startsWith(".."))
              attrs.src = `https://raw.githubusercontent.com/Motoki0705/tennis-lab/main/${repoPath}`;
          }
        }
        return { tagName: "img", attribs: attrs };
      },
      a: (_tag, attrs) => {
        const href = attrs.href ?? "";
        if (!/^(?:[a-z]+:|\/|#)/i.test(href)) {
          const [local, fragment] = href.split("#");
          const target = targets.get(path.resolve(path.dirname(file), local));
          if (target) attrs.href = target;
          else {
            const relative = path.relative(
              path.resolve(KNOWLEDGE_DIR, ".."),
              path.resolve(path.dirname(file), local),
            );
            if (!relative.startsWith(".."))
              attrs.href = `https://github.com/Motoki0705/tennis-lab/blob/main/${relative}${fragment ? `#${fragment}` : ""}`;
          }
        }
        return { tagName: "a", attribs: attrs };
      },
    },
  });
}
