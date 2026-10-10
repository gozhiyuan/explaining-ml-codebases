#!/usr/bin/env python3
"""Render a state trace to SVG keyframes and a standalone HTML stepper.

Stdlib only. See visual-guide.md for the JSON contract. All frames are complete
snapshots; durations are presentation timing, never performance measurements.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
import textwrap
from pathlib import Path

TONES = {'blue': ('#eaf2ff', '#2563eb'), 'green': ('#e8f6ef', '#168052'),
         'amber': ('#fff4d9', '#a66a08'), 'purple': ('#f1ebff', '#7541b8'),
         'gray': ('#f0f3f7', '#526174')}


def validate(trace):
    if not isinstance(trace, dict) or not isinstance(trace.get('title'), str):
        raise ValueError('trace needs a title string')
    if trace.get('evidence_status') not in ('illustrative', 'fixture', 'measured'):
        raise ValueError('evidence_status must be illustrative, fixture, or measured')
    if not isinstance(trace.get('description', ''), str):
        raise ValueError('description must be a string')
    if not isinstance(trace.get('frames'), list) or not trace['frames']:
        raise ValueError('frames must be a non-empty array')
    for i, frame in enumerate(trace['frames']):
        if not isinstance(frame, dict) or not isinstance(frame.get('title'), str):
            raise ValueError(f'frame {i}: needs a title string')
        duration = frame.get('duration', 3)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'frame {i}: duration must be finite and positive')
        if not isinstance(frame.get('caption', ''), str):
            raise ValueError(f'frame {i}: caption must be a string')
        panels = frame.get('panels')
        if not isinstance(panels, list) or not 1 <= len(panels) <= 3:
            raise ValueError(f'frame {i}: use 1-3 panels; split dense scenes')
        for panel in panels:
            if not isinstance(panel, dict) or not isinstance(panel.get('label'), str):
                raise ValueError(f'frame {i}: panel needs a label string')
            if panel.get('tone', 'blue') not in TONES:
                raise ValueError(f'frame {i}: unknown panel tone')
            if not isinstance(panel.get('items', []), list) or not all(isinstance(s, str) for s in panel.get('items', [])):
                raise ValueError(f'frame {i}: panel items must be strings')
            if not isinstance(panel.get('tokens', []), list) or not all(isinstance(s, (str, int)) and not isinstance(s, bool) for s in panel.get('tokens', [])):
                raise ValueError(f'frame {i}: tokens must be strings or integers')
    return trace


def lines(text, width):
    return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or ['']


def frame_svg(trace, frame, index):
    count = len(frame['panels'])
    width = (896 - (count - 1) * 20) / count
    columns = max(8, int((width - 40) / 11.5))
    prepared = []
    for panel in frame['panels']:
        label = lines(panel['label'], columns)
        body = [line for item in panel.get('items', []) for line in lines(item, columns)]
        tokens = panel.get('tokens', [])
        token_cols = max(1, int((width - 40) / 58))
        height = max(190, 40 + len(label) * 28 + len(body) * 28 + math.ceil(len(tokens) / token_cols) * 42 + 24)
        prepared.append((panel, label, body, tokens, token_cols, height))
    panel_height = max(p[-1] for p in prepared)
    title_lines = lines(frame['title'], 52)
    panel_y = 100 + len(title_lines) * 34
    captions = lines(frame.get('caption', ''), 85)
    height = panel_y + panel_height + 40 + len(captions) * 25 + 40
    escape = lambda s: html.escape(str(s), quote=True)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" role="img" aria-label="{escape(frame["title"])}">',
           f'<title>{escape(frame["title"])}</title>',
           f'<desc>{escape(frame.get("caption", ""))}</desc>',
           '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172b45}.body{font-size:22px}.label{font-size:22px;font-weight:700}.token{font:20px monospace;text-anchor:middle}</style>',
           f'<rect width="960" height="{height}" fill="#fff"/>',
           f'<text x="32" y="35" font-size="15" fill="#526174">STEP {index + 1} / {len(trace["frames"])} · {escape(trace["evidence_status"])} state · illustrative timing</text>']
    for i, line in enumerate(title_lines):
        svg.append(f'<text x="32" y="{78 + i * 34}" font-size="28" font-weight="700">{escape(line)}</text>')
    for i, (panel, label, body, tokens, token_cols, _) in enumerate(prepared):
        x = 32 + i * (width + 20)
        fill, accent = TONES[panel.get('tone', 'blue')]
        svg.append(f'<rect x="{x}" y="{panel_y}" width="{width}" height="{panel_height}" rx="14" fill="{fill}" stroke="{accent}" stroke-width="1.5"/>')
        y = panel_y + 34
        for line in label:
            svg.append(f'<text x="{x + 20}" y="{y}" class="label">{escape(line)}</text>')
            y += 28
        y += 12
        for line in body:
            svg.append(f'<text x="{x + 20}" y="{y}" class="body">{escape(line)}</text>')
            y += 28
        y += 12
        for j, token in enumerate(tokens):
            tx, ty = x + 20 + (j % token_cols) * 58, y + (j // token_cols) * 42
            if len(str(token)) > 4:
                raise ValueError('token labels must be at most 4 characters; put long values in items')
            svg.append(f'<rect x="{tx}" y="{ty}" width="50" height="32" rx="6" fill="#fff" stroke="{accent}"/>')
            svg.append(f'<text x="{tx + 25}" y="{ty + 22}" class="token">{escape(token)}</text>')
    for i, line in enumerate(captions):
        svg.append(f'<text x="32" y="{panel_y + panel_height + 36 + i * 25}" class="body">{escape(line)}</text>')
    svg.append('</svg>')
    return '\n'.join(svg), height


TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;padding:16px;font:16px/1.5 Arial,Helvetica,sans-serif;color:#172b45;background:#fff}
h1{font-size:22px;margin:0 0 8px}p{margin:8px 0;color:#526174}.scene{width:100%;max-width:960px;margin:auto}.scene svg{display:block;width:100%;height:auto}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px auto;max-width:960px}
button{padding:8px 12px;border:1px solid #bac7d5;border-radius:6px;background:#f4f7fb;color:#172b45;font:inherit;cursor:pointer}button:disabled{opacity:.45}
input{flex:1;min-width:90px}details{max-width:960px;margin:auto}summary{cursor:pointer}
details svg{display:block;width:100%;height:auto}.mobile-hint{display:none}
@media(max-width:600px){body{padding:8px}.scene{overflow-x:auto}.scene svg{min-width:700px}h1{font-size:18px}.mobile-hint{display:block;font-size:14px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><h1>__TITLE__</h1><p>__DESCRIPTION__</p>
<p class="mobile-hint">Scroll the figure horizontally to see all panels, or expand “Read all steps” below.</p>
<div class="scene" id="scene" tabindex="0" aria-label="State figure; scroll horizontally on small screens">__FIRST__</div>
<div class="controls"><button id="prev" aria-label="Previous step">Previous</button><button id="play">Play</button><button id="next" aria-label="Next step">Next</button>
<label for="step">Step</label><input id="step" type="range" min="0" max="__MAX__" value="0"><span id="status" aria-live="polite"></span></div>
<details><summary>Read all steps</summary>__FALLBACK__</details>
<script id="trace-data" type="application/json">__DATA__</script><script>
(()=>{const data=JSON.parse(document.getElementById('trace-data').textContent), scene=document.getElementById('scene'),slider=document.getElementById('step'),play=document.getElementById('play');
let index=0,timer=null;const duration=data.frames.reduce((s,f)=>s+f.duration,0);
function stop(){clearTimeout(timer);timer=null;play.textContent='Play'}
function show(i){index=Math.max(0,Math.min(data.frames.length-1,i));scene.innerHTML=data.frames[index].svg;slider.value=index;document.getElementById('status').textContent=`${index+1} / ${data.frames.length}`;document.getElementById('prev').disabled=index===0;document.getElementById('next').disabled=index===data.frames.length-1}
function advance(){if(index===data.frames.length-1){stop();return}show(index+1);timer=setTimeout(advance,data.frames[index].duration*1000)}
document.getElementById('prev').onclick=()=>{stop();show(index-1)};document.getElementById('next').onclick=()=>{stop();show(index+1)};slider.oninput=()=>{stop();show(Number(slider.value))};
play.onclick=()=>{if(timer){stop();return}if(index===data.frames.length-1)show(0);play.textContent='Pause';timer=setTimeout(advance,data.frames[index].duration*1000)};
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});
window.explainTrace={duration,frameCount:data.frames.length,height:data.height,seek(t){stop();let elapsed=0;for(let i=0;i<data.frames.length;i++){elapsed+=data.frames[i].duration;if(t<elapsed||i===data.frames.length-1){show(i);return i}}}};
show(0);})();</script></body></html>'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace', type=Path)
    ap.add_argument('-o', '--out', required=True, type=Path)
    ap.add_argument('--frames-dir', type=Path, help='also save SVG keyframes and poster.svg')
    args = ap.parse_args(argv)
    try:
        trace = validate(json.loads(args.trace.read_text()))
        frames = []
        heights = []
        for i, frame in enumerate(trace['frames']):
            svg, height = frame_svg(trace, frame, i)
            frames.append({'svg': svg, 'duration': frame.get('duration', 3)})
            heights.append(height)
        # Keep every frame at the same displayed size for video export.
        height = max(heights)
        for f, h in zip(frames, heights):
            f['svg'] = f['svg'].replace(f'viewBox="0 0 960 {h}"', f'viewBox="0 0 960 {height}"')
        if args.frames_dir:
            args.frames_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                (args.frames_dir / f'frame-{i:03d}.svg').write_text(f['svg'])
            (args.frames_dir / 'poster.svg').write_text(frames[0]['svg'])
        fallback = ''.join(f'<h2>{html.escape(f["title"])}</h2><p>{html.escape(f.get("caption", ""))}</p>{rendered["svg"]}' for f, rendered in zip(trace['frames'], frames))
        data = json.dumps({'frames': frames, 'height': height}, ensure_ascii=False).replace('</', '<\\/')
        page = TEMPLATE.replace('__TITLE__', html.escape(trace['title'])).replace('__DESCRIPTION__', html.escape(trace.get('description', '')))
        page = page.replace('__MAX__', str(len(frames) - 1)).replace('__FIRST__', frames[0]['svg']).replace('__FALLBACK__', fallback).replace('__DATA__', data)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(page)
        print(f'wrote {args.out}: {len(frames)} states; standalone HTML, no libraries required')
        return 0
    except (ValueError, OSError, TypeError) as e:
        print(f'ERROR {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
