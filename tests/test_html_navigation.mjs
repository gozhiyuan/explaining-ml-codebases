// Optional real-browser regression checks. Requires Playwright and an available browser.
// Run: node tests/test_html_navigation.mjs (set PLAYWRIGHT_CHANNEL=chrome to use installed Chrome).
import { createRequire } from 'node:module';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync, copyFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
const { chromium } = createRequire(import.meta.url)('playwright');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const temp = mkdtempSync(join(tmpdir(), 'explain-navigation-'));
const args = process.argv.slice(2), artifactIndex = args.indexOf('--artifacts');
const artifacts = artifactIndex >= 0 ? resolve(args[artifactIndex + 1]) : null;
const docs = join(temp, 'docs'); mkdirSync(docs);
writeFileSync(join(docs, 'index.md'), '# Index\n\n## Overview\n\nIndex overview.\n\n[Training](train.md)\n');
writeFileSync(join(docs, 'model.md'), '# Model\n\n## Contents\n\n[Overview](#overview)\n\n## Overview\n\nModel overview.\n');
copyFileSync(join(root, 'examples/vllm/assets/engine-step.svg'), join(docs, 'engine-step.svg'));
writeFileSync(join(docs, 'train.md'), `# Training

## Contents

[Overview](#overview)
[Café](#caf%C3%A9)
[Model](model.md#overview)

<a href="#overview">Raw local anchor</a>

[TOC]

## Overview

Training overview stays in this document.

![Original engine ownership figure](engine-step.svg)

#### Repeated

A deeper heading consumes a slug.

## Repeated

First top-level repeated heading.

## Repeated-1

A title that collides with a generated suffix.

## Repeated

Second top-level repeated heading.

## Café

Unicode heading.

## Process map

\`\`\`mermaid
flowchart LR
    a["Input"] --> b["Prepare"] --> c["Choose"] --> d["Allocate"] --> e["Compute"] --> f["Collect"] --> g["Update"] --> h["Output"]
\`\`\`
`);
const output = join(temp, 'explain.html');
execFileSync(process.env.PYTHON || 'python3', [join(root, 'scripts/render_html.py'), docs, '-o', output]);
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, ignoreHTTPSErrors: true });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto(pathToFileURL(output).href);
  await page.locator('#content h1').filter({ hasText: 'Index' }).waitFor({ timeout: 60000 });
  await page.locator('#nav a.doc').filter({ hasText: 'Training' }).click();
  await page.locator('.mermaid svg').waitFor({ timeout: 60000 });
  await page.locator('#content a').filter({ hasText: /^Overview$/ }).first().click();
  await page.waitForFunction(() => location.hash === '#/train.md#overview');
  assert.equal(await page.locator('#content h1').textContent(), 'Training');
  await page.locator('.mermaid svg').waitFor();
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.locator('#nav a.h2').filter({ hasText: /^Overview$/ }).click();
  await page.waitForFunction(() => { const y = document.getElementById('overview').getBoundingClientRect().top; return y >= 0 && y < 100; });
  await page.locator('#content a').filter({ hasText: /^Café$/ }).first().click();
  await page.waitForFunction(() => location.hash === '#/train.md#caf%C3%A9');
  assert.equal(await page.locator('#content #café').count(), 1);
  await page.locator('#content a').filter({ hasText: 'Raw local anchor' }).click();
  await page.waitForFunction(() => location.hash === '#/train.md#overview');
  await page.locator('#content a').filter({ hasText: /^Model$/ }).click();
  await page.locator('#content h1').filter({ hasText: 'Model' }).waitFor();
  await page.goBack();
  await page.locator('#content h1').filter({ hasText: 'Training' }).waitFor();
  assert.equal(new URL(page.url()).hash, '#/train.md#overview');
  await page.locator('.mermaid svg').waitFor();
  const ids = await page.locator('#content h1, #content h2, #content h3, #content h4').evaluateAll(nodes => nodes.map(n => n.id));
  assert.equal(new Set(ids).size, ids.length, 'Every rendered heading must have a unique ID');
  const tocTargets = await page.locator('#nav a.h2, #nav a.h3').evaluateAll(nodes => nodes.map(n => decodeURIComponent(n.hash.split('#').slice(2).join('#'))));
  for (const id of tocTargets) assert(ids.includes(id), `Sidebar target ${id} must exist`);
  assert(ids.includes('repeated-2'));
  const diagram = page.locator('.mermaid svg');
  await page.getByRole('button', { name: 'Fit', exact: true }).click();
  const fitted = await diagram.boundingBox(); assert(fitted.height <= 501);
  await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
  assert((await diagram.boundingBox()).width > fitted.width);
  await page.getByRole('button', { name: 'Fit', exact: true }).click();
  assert(Math.abs((await diagram.boundingBox()).width - fitted.width) < 2);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Fit', exact: true }).click();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Diagram must not widen the mobile page');
  assert((await diagram.boundingBox()).height <= 501);
  if (artifacts) {
    mkdirSync(artifacts, { recursive: true });
    for (let i = 0; i < 3; i++) await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
    await page.locator('.diagram-tools').scrollIntoViewIfNeeded();
    await page.screenshot({ path: join(artifacts, 'mobile-diagram.png') });
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.getByRole('button', { name: 'Fit', exact: true }).click();
    await page.locator('#content img').scrollIntoViewIfNeeded();
    await page.screenshot({ path: join(artifacts, 'desktop-overview.png') });
    await page.locator('#content img').screenshot({ path: join(artifacts, 'engine-step.png') });
  }
  assert.deepEqual(errors, []);
  console.log('PASS: local/cross-doc TOCs, Unicode anchors, back navigation, unique heading IDs, diagram fit/zoom and mobile width');
} finally {
  await browser.close(); rmSync(temp, { recursive: true, force: true });
}
