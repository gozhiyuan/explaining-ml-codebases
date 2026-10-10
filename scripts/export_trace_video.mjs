#!/usr/bin/env node
/** Export a render_trace.py player to MP4. Optional: playwright + Chromium,
 * ffmpeg on PATH. NODE_PATH may point to an existing package directory.
 * Usage: node export_trace_video.mjs trace.html -o trace.mp4 [--fps 12]
 *        [--browser-channel chrome]
 */
import { createRequire } from 'node:module';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

function run(command, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let error = '';
    proc.stderr.on('data', chunk => { error = (error + chunk).slice(-4000); });
    proc.on('error', reject);
    proc.on('close', code => code === 0 ? resolve() : reject(new Error(`${command} exited ${code}: ${error}`)));
  });
}

let browser, scratch;
try {
  const args = process.argv.slice(2);
  const input = args.shift();
  let output, fps = 12, channel;
  while (args.length) {
    const arg = args.shift();
    if (arg === '-o') output = args.shift();
    else if (arg === '--fps') fps = Number(args.shift());
    else if (arg === '--browser-channel') {
      channel = args.shift();
      if (!['chrome', 'msedge'].includes(channel)) throw new Error('--browser-channel must be chrome or msedge');
    }
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!input || !output || !Number.isInteger(fps) || fps < 1 || fps > 60)
    throw new Error('Usage: node export_trace_video.mjs trace.html -o trace.mp4 [--fps 12]; fps must be 1-60');
  if (path.extname(output).toLowerCase() !== '.mp4' || path.resolve(input) === path.resolve(output))
    throw new Error('Output must be a separate .mp4 file');
  await run('ffmpeg', ['-version']);
  const require = createRequire(import.meta.url);
  let chromium;
  try { ({ chromium } = require('playwright')); }
  catch { throw new Error('Install optional playwright and Chromium, or set NODE_PATH to an existing package directory; see visual-guide.md'); }
  browser = await chromium.launch({ headless: true, ...(channel ? { channel } : {}) });
  const page = await browser.newPage({ viewport: { width: 992, height: 900 }, reducedMotion: 'reduce' });
  await page.goto(pathToFileURL(path.resolve(input)).href);
  const trace = await page.evaluate(() => window.explainTrace && ({ duration: window.explainTrace.duration, height: window.explainTrace.height }));
  if (!trace || !Number.isFinite(trace.duration) || trace.duration <= 0)
    throw new Error('Input must be a standalone player produced by render_trace.py');
  await page.setViewportSize({ width: 992, height: Math.ceil(trace.height) + 220 });
  await page.evaluate(() => document.fonts.ready);
  scratch = await mkdtemp(path.join(tmpdir(), 'explain-trace-'));
  const count = Math.ceil(trace.duration * fps);
  console.log(`Rendering ${count} frames (${trace.duration}s at ${fps} fps)`);
  for (let i = 0; i < count; i++) {
    await page.evaluate(t => window.explainTrace.seek(t), i / fps);
    await page.locator('#scene').screenshot({ path: path.join(scratch, `frame-${String(i).padStart(6, '0')}.png`) });
  }
  await mkdir(path.dirname(path.resolve(output)), { recursive: true });
  // Refuse to overwrite an existing export; the caller can choose a new name.
  await run('ffmpeg', ['-n', '-framerate', String(fps), '-i', path.join(scratch, 'frame-%06d.png'),
    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output]);
  console.log(`Wrote ${output}`);
} catch (e) {
  console.error(`ERROR ${e.message}`);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (scratch) await rm(scratch, { recursive: true, force: true });
}
