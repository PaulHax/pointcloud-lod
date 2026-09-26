#!/usr/bin/env node
/**
 * Where a V8 CPU profile spent its time: the functions with the most self
 * time, and the most inclusive time, over the whole profile.
 *
 *   node scripts/summarizeProfile.mjs <file.cpuprofile> [rows]
 *
 * Self time is what a function did itself; inclusive time adds everything it
 * called. A function called from itself is counted once per sample, so
 * recursion does not inflate its inclusive time.
 */
import { readFileSync } from "node:fs";

const [path, rowsArgument = "30"] = process.argv.slice(2);
if (!path) {
  console.error("usage: summarizeProfile.mjs <file.cpuprofile> [rows]");
  process.exit(2);
}
const rows = Number(rowsArgument);
const profile = JSON.parse(readFileSync(path, "utf8"));

const nodes = new Map(profile.nodes.map((node) => [node.id, node]));
const parents = new Map();
for (const node of profile.nodes) {
  for (const child of node.children ?? []) parents.set(child, node.id);
}
const labelOf = (node) => {
  const { functionName, url, lineNumber } = node.callFrame;
  const file = url ? url.slice(url.lastIndexOf("/") + 1) : "";
  return `${functionName || "(anonymous)"} ${file}:${lineNumber + 1}`;
};

// Each sample is charged the interval until the next one.
const selfMs = new Map();
const totalMs = new Map();
const deltas = profile.timeDeltas ?? [];
profile.samples.forEach((id, index) => {
  const ms = (deltas[index + 1] ?? 0) / 1000;
  const leaf = labelOf(nodes.get(id));
  selfMs.set(leaf, (selfMs.get(leaf) ?? 0) + ms);
  const seen = new Set();
  for (let at = id; at !== undefined; at = parents.get(at)) {
    const label = labelOf(nodes.get(at));
    if (seen.has(label)) continue;
    seen.add(label);
    totalMs.set(label, (totalMs.get(label) ?? 0) + ms);
  }
});

const print = (title, map) => {
  console.log(`--- ${title} ---`);
  for (const [label, ms] of [...map]
    .sort((a, b) => b[1] - a[1])
    .slice(0, rows)) {
    console.log(`${ms.toFixed(0).padStart(8)} ms  ${label}`);
  }
};
const spanMs = (profile.endTime - profile.startTime) / 1000;
console.log(
  `profile span ${spanMs.toFixed(0)} ms, ${profile.samples.length} samples`,
);
print("self", selfMs);
print("inclusive", totalMs);
