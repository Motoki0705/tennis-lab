export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function isTypingInField(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}

export function formatAssistMetaSummary(meta: {
  model_type: string;
  device: string;
  score_threshold: number;
  batch_size: number;
  max_disp: number;
} | null): string {
  if (!meta) return "assist not configured";
  return `model=${meta.model_type}, device=${meta.device}, score=${meta.score_threshold}, batch=${meta.batch_size}, max_disp=${meta.max_disp}`;
}
