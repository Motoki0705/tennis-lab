#!/usr/bin/env node
/** Install, inspect, or run only the dsh package beside this entrypoint. */

import { spawn, spawnSync } from "node:child_process";
import {
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const runtimeDir = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(runtimeDir, "package.json"), "utf8"));
const expectedVersion = manifest.dependencies["@deepseek-ai/dsh"];
const nodeRange = manifest.engines.node;
const installHint = 'Run "node tools/dsh/runtime.mjs install" in the runtime-owning worktree.';

const usage = `Usage: node tools/dsh/runtime.mjs <command>
  install              Install package-lock.json with npm ci; check the CLI version.
  check                Print the selected Node/dsh paths and versions as JSON.
  exec -- <dsh args>   Check the runtime, then pass args/stdin/stdout/stderr to dsh.
`;

/** Match ^22.19.0 || >=24.0.0, including semver's prerelease exclusion. */
export function requireSupportedNode(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  const major = match ? Number(match[1]) : 0;
  const minor = match ? Number(match[2]) : 0;
  if (!match || !((major === 22 && minor >= 19) || major >= 24)) {
    throw new Error(`Unsupported Node.js ${version}; required ${nodeRange}.`);
  }
}

function readInstalledRuntime() {
  const packageDir = join(runtimeDir, "node_modules", "@deepseek-ai", "dsh");
  let installed;
  try {
    installed = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"));
  } catch (error) {
    throw new Error(`Cannot read the local dsh installation at ${packageDir}. ${installHint}`, {
      cause: error,
    });
  }
  if (installed.name !== "@deepseek-ai/dsh" || installed.version !== expectedVersion) {
    throw new Error(
      `dsh version mismatch at ${packageDir}: expected @deepseek-ai/dsh@${expectedVersion}, ` +
      `found ${installed.name}@${installed.version}. ${installHint}`,
    );
  }
  if (installed.bin?.dsh !== "lib/bin.js") {
    throw new Error(`Unexpected dsh bin entry at ${packageDir}. ${installHint}`);
  }
  try {
    return realpathSync(join(packageDir, "lib", "bin.js"));
  } catch (error) {
    throw new Error(`Missing local dsh executable at ${packageDir}/lib/bin.js. ${installHint}`, {
      cause: error,
    });
  }
}

function checkRuntime() {
  const executable = readInstalledRuntime();
  // -V must not discover credentials or .env files in the caller's workspace.
  const checkHome = mkdtempSync(join(tmpdir(), "dsh-runtime-check-"));
  const checkEnv = { ...process.env, DSH_HOME: checkHome, DSH_TELEMETRY_DISABLED: "1" };
  delete checkEnv.DEEPSEEK_API_KEY;
  let result;
  try {
    result = spawnSync(process.execPath, [executable, "-V"], {
      cwd: checkHome,
      env: checkEnv,
      encoding: "utf8",
      timeout: 15_000,
      killSignal: "SIGKILL",
      maxBuffer: 1024 * 1024,
    });
  } finally {
    rmSync(checkHome, { recursive: true, force: true });
  }
  if (result.error || result.status !== 0) {
    // Do not reproduce arbitrary CLI output that might contain environment data.
    const reason = result.error?.code ?? result.signal ?? `exit ${result.status}`;
    throw new Error(`Local dsh -V failed (${reason}) at ${executable}. ${installHint}`);
  }
  const version = result.stdout.trim();
  if (version !== expectedVersion) {
    throw new Error(
      `dsh CLI version mismatch at ${executable}: expected ${expectedVersion}, ` +
      `received an unexpected version. ${installHint}`,
    );
  }
  return {
    node_executable: realpathSync(process.execPath),
    node_version: process.versions.node,
    node_requirement: nodeRange,
    dsh_executable: executable,
    dsh_version: version,
  };
}

function install() {
  // npm may itself have an /usr/bin/env node shebang. Use the same Node as below.
  const env = {
    ...process.env,
    PATH: `${dirname(process.execPath)}${delimiter}${process.env.PATH ?? ""}`,
    DSH_TELEMETRY_DISABLED: "1",
  };
  delete env.DEEPSEEK_API_KEY;
  const result = spawnSync(
    "npm",
    ["ci", "--no-audit", "--no-fund"],
    { cwd: runtimeDir, env, stdio: "inherit" },
  );
  if (result.error || result.status !== 0) {
    const reason = result.error?.code ?? result.signal ?? `exit ${result.status}`;
    throw new Error(`npm ci failed (${reason}); the dsh runtime was not verified.`);
  }
}

async function run(executable, args) {
  // Keep the caller's cwd and standard streams. Signal/exit status belong to dsh.
  const child = spawn(process.execPath, [executable, ...args], { stdio: "inherit" });
  const signals = ["SIGINT", "SIGTERM", "SIGHUP"];
  const handlers = signals.map((signal) => {
    const handler = () => child.kill(signal);
    process.on(signal, handler);
    return handler;
  });
  const cleanup = () => {
    for (let i = 0; i < signals.length; i++) process.off(signals[i], handlers[i]);
  };
  await new Promise((resolveRun, rejectRun) => {
    child.once("error", (error) => {
      cleanup();
      rejectRun(error);
    });
    child.once("exit", (code, signal) => {
      cleanup();
      if (signal) process.kill(process.pid, signal);
      else process.exitCode = code;
      resolveRun();
    });
  });
}

export async function main(args = process.argv.slice(2)) {
  if (args.length === 1 && ["--help", "-h"].includes(args[0])) {
    process.stdout.write(usage);
    return;
  }
  const [command, ...rest] = args;
  if (
    !["install", "check", "exec"].includes(command) ||
    (command !== "exec" && rest.length !== 0) ||
    (command === "exec" && (rest[0] !== "--" || rest.length < 2))
  ) {
    process.stderr.write(usage);
    process.exitCode = 2;
    return;
  }
  requireSupportedNode(process.versions.node);
  if (command === "install") install();
  const runtime = checkRuntime();
  if (command === "exec") await run(runtime.dsh_executable, rest.slice(1));
  else process.stdout.write(`${JSON.stringify(runtime)}\n`);
}

if (process.argv[1] && realpathSync(resolve(process.argv[1])) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    process.stderr.write(`dsh runtime: ${error.message}\n`);
    process.exitCode = 1;
  });
}
