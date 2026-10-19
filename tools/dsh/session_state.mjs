#!/usr/bin/env node
/** Read a pinned persisted session without booting an agent or calling an API. */

import { join } from "node:path";
import { Context } from "@deepseek-ai/cordis";
import JsonlSessionPersistence from "@deepseek-ai/dsh-session-persistence-jsonl";
import { Session, SessionId } from "@deepseek-ai/dsh-session";
import { readColdSessionLog } from "@deepseek-ai/dsh-session-query";
import { currentSessionMessageProjections } from "@deepseek-ai/dsh-session-format-catalog/message-projections";

try {
  const [home, identity] = process.argv.slice(2);
  if (!home || !identity || process.argv.length !== 4) throw new Error();
  const context = new Context();
  const persistence = new JsonlSessionPersistence(context, {
    root: join(home, "sessions"), compression: "zstd",
  });
  const id = SessionId(identity);
  const cold = await readColdSessionLog(persistence, id);
  // Replay the full log through the pinned semantic projections too. No
  // hand-written zstd reader, old-format migration, or write handle is used.
  Session.create(id, cold.events, cold.header, cold.inheritedEventCount,
    currentSessionMessageProjections);
  let mode = null, approval = null, preset = null, lastTurn = 0;
  let agentPreset = cold.header.agentPreset;
  for (const event of cold.events) {
    if (event.type === "sandbox/mode") mode = event.data.mode;
    if (event.type === "approval/policy") approval = event.data.policy;
    if (event.type === "permission/preset") preset = event.data.preset;
    if (event.type === "agent-preset/selected") agentPreset = event.data.preset;
    if (event.type === "turn/start" || event.type === "turn/end") {
      if (!Number.isInteger(event.data.turn) || event.data.turn < 1) throw new Error();
      lastTurn = Math.max(lastTurn, event.data.turn);
    }
  }
  if (cold.header.id !== identity || cold.header.parentSession !== undefined
      || cold.header.origin === "subagent" || agentPreset !== undefined
      || typeof cold.header.cwd !== "string") throw new Error();
  process.stdout.write(JSON.stringify({
    session_id: identity, cwd: cold.header.cwd, mode, approval, preset,
    last_turn: lastTurn,
  }) + "\n");
} catch {
  // Backend/provider-controlled exception messages are never public output.
  process.stderr.write("Cannot inspect the pinned persisted session; preserve its private log.\n");
  process.exitCode = 1;
}
