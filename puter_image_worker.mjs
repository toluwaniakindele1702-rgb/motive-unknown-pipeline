#!/usr/bin/env node
/**
 * Relic Loop Puter image worker.
 *
 * Uses the user's Puter account auth token in PUTER_AUTH_TOKEN.
 * Puter routes the request to the selected image model; no separate
 * Black Forest Labs/OpenAI/Replicate API key is required.
 */

const fs = require("fs/promises");
const path = require("path");

async function loadPuter() {
  try {
    const mod = await import("@heyputer/puter.js/src/init.cjs");
    const init = mod.init || mod.default?.init;
    if (!init) throw new Error("Puter.js init() was not exported.");
    const token = process.env.PUTER_AUTH_TOKEN?.trim();
    if (!token) throw new Error("PUTER_AUTH_TOKEN is not configured.");
    return init(token);
  } catch (err) {
    throw new Error(`Could not initialize Puter.js: ${err.message || err}`);
  }
}

function dataUriToBuffer(src) {
  const match = /^data:([^;]+);base64,(.*)$/s.exec(src);
  if (!match) return null;
  return Buffer.from(match[2], "base64");
}

async function downloadImage(src) {
  const data = dataUriToBuffer(src);
  if (data) return data;
  if (!/^https?:\\/\\//i.test(src)) {
    throw new Error("Puter returned an unsupported image source.");
  }
  const response = await fetch(src);
  if (!response.ok) {
    throw new Error(`Image download failed with HTTP ${response.status}`);
  }
  return Buffer.from(await response.arrayBuffer());
}

async function main() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const request = JSON.parse(Buffer.concat(chunks).toString("utf8"));

  if (request.cmd !== "generate") {
    throw new Error(`Unknown command: ${request.cmd}`);
  }

  const puter = await loadPuter();
  const model = request.model || "black-forest-labs/flux-2-pro";
  const prompt = String(request.prompt || "").trim();
  const outputPath = path.resolve(String(request.output_path));

  if (!prompt) throw new Error("Image prompt is empty.");

  console.log(`[PUTER] model=${model} seed=${request.seed ?? "default"}`);

  const image = await puter.ai.txt2img(prompt, {
    model,
    ratio: { w: 16, h: 9 },
    output_megapixels: request.output_megapixels || "1",
    response_format: "jpg",
    safety_tolerance: 2,
    prompt_upsampling: false,
  });

  const src = image?.src || (typeof image === "string" ? image : String(image));
  const bytes = await downloadImage(src);

  if (bytes.length < 10000) {
    throw new Error("Puter returned an unexpectedly small image.");
  }

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, bytes);
  console.log(`[PUTER] wrote ${outputPath} (${bytes.length} bytes)`);
}

main().catch((err) => {
  console.error(`[PUTER ERROR] ${err.stack || err.message || err}`);
  process.exit(1);
});
