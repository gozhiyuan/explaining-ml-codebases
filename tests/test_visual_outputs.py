"""Behavior checks for portable exports, shared snapshots, and evidence retention.

Run: python -m unittest discover -s tests -v
"""
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


class Outputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / 'docs'
        self.docs.mkdir()

    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *map(str, args)], capture_output=True, text=True)

    def media_fixture(self):
        media = self.docs / 'assets'
        media.mkdir()
        (media / 'poster.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        (media / 'clip.mp4').write_bytes(b'fixture-video')
        (media / 'player.html').write_text('<html><button>Next</button></html>')
        (self.docs / 'evidence.json').write_text('{"schema_version":1}')
        (self.docs / 'index.md').write_text('''# Media export

![Poster](assets/poster.svg)

<video controls
  poster='assets/poster.svg'>
  <source src='assets/clip.mp4' type='video/mp4'>
  <a href='assets/clip.mp4'>Download</a>
</video>

<iframe title="State player" src='assets/player.html' sandbox="allow-scripts"></iframe>

[Open player](assets/player.html)
[Download video](assets/clip.mp4)
[Evidence](evidence.json)

```html
<video src="not-a-real-file.mp4"></video>
```
''')

    def payload(self, out):
        return json.loads(re.search(r'<script id="docs-data" type="application/json">(.*?)</script>', out.read_text(), re.S).group(1))

    def test_auto_export_survives_relocation_and_deduplicates(self):
        self.media_fixture()
        out = self.root / 'export' / 'reading view.html'
        result = self.run_script('render_html.py', self.docs, '-o', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(out.with_suffix('.assets').iterdir())), 3)
        md = self.payload(out)[0]['md']
        self.assertIn('data:image/svg+xml;base64,', md)
        self.assertIn('not-a-real-file.mp4', md)
        shutil.copytree(out.parent, self.root / 'moved')
        shutil.rmtree(self.docs)
        shutil.rmtree(out.parent)
        moved = self.root / 'moved' / out.name
        refs = re.findall(r'(reading%20view\.assets/[^\s"\x27)]+)', self.payload(moved)[0]['md'])
        self.assertGreaterEqual(len(refs), 5)
        for ref in refs:
            self.assertTrue((moved.parent / unquote(ref)).is_file(), ref)
        self.assertNotIn(str(self.root), moved.read_text())

    def test_inline_export_embeds_video_poster_and_player(self):
        self.media_fixture()
        out = self.root / 'inline.html'
        result = self.run_script('render_html.py', self.docs, '-o', out, '--media-mode', 'inline')
        self.assertEqual(result.returncode, 0, result.stderr)
        md = self.payload(out)[0]['md']
        self.assertIn('data:video/mp4;base64,' + base64.b64encode(b'fixture-video').decode(), md)
        self.assertIn('data:text/html;base64,', md)
        self.assertIn('data:image/svg+xml;base64,', md)
        self.assertIn('data:application/json;base64,', md)
        self.assertFalse(out.with_suffix('.assets').exists())

    def test_html_only_player_restored_and_portable_without_rewriting_code(self):
        self.media_fixture()
        block = '<!-- explain-html-only\n<iframe src="assets/player.html" title="Trace"></iframe>\n-->'
        source = '# Shared Markdown\n\n' + block + '\n\n```html\n' + block + '\n```\n'
        (self.docs / 'index.md').write_text(source)
        out = self.root / 'reading.html'
        result = self.run_script('render_html.py', self.docs, '-o', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = self.payload(out)[0]['md']
        prose, code = rendered.split('```html', 1)
        self.assertNotIn('explain-html-only', prose)
        self.assertIn('<iframe src="reading.assets/player-', prose)
        self.assertIn(block, code)
        packaged = list(out.with_suffix('.assets').glob('player-*.html'))
        self.assertEqual(len(packaged), 1)
        self.assertEqual(packaged[0].read_text(), (self.docs / 'assets/player.html').read_text())
        self.assertEqual((self.docs / 'index.md').read_text(), source)

    def test_bundle_handles_same_named_different_assets(self):
        (self.docs / 'a').mkdir(); (self.docs / 'b').mkdir()
        (self.docs / 'a' / 'same.svg').write_text('<svg>A</svg>')
        (self.docs / 'b' / 'same.svg').write_text('<svg>B</svg>')
        (self.docs / 'index.md').write_text('# Figures\n\n![A](a/same.svg)\n![B](b/same.svg)\n')
        out = self.root / 'bundle.html'
        result = self.run_script('render_html.py', self.docs, '-o', out, '--media-mode', 'bundle')
        self.assertEqual(result.returncode, 0, result.stderr)
        files = list(out.with_suffix('.assets').iterdir())
        self.assertEqual({p.read_text() for p in files}, {'<svg>A</svg>', '<svg>B</svg>'})

    def test_missing_media_and_inline_size_limit_fail_without_page(self):
        out = self.root / 'failed.html'
        (self.docs / 'index.md').write_text('# Missing\n\n<video src="missing.mp4"></video>\n')
        result = self.run_script('render_html.py', self.docs, '-o', out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('media not found', result.stderr)
        self.assertFalse(out.exists())
        (self.docs / 'large.mp4').write_bytes(b'x' * 2048)
        (self.docs / 'index.md').write_text('# Large\n\n<video src="large.mp4"></video>\n')
        result = self.run_script('render_html.py', self.docs, '-o', out, '--media-mode', 'inline', '--max-inline-mb', '.001')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exceeds', result.stderr)
        self.assertFalse(out.exists())

    def test_remote_media_is_preserved(self):
        (self.docs / 'index.md').write_text('# Remote\n\n<video src="https://example.org/x.mp4" poster="//example.org/x.png"></video>\n')
        out = self.root / 'remote.html'
        result = self.run_script('render_html.py', self.docs, '-o', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('https://example.org/x.mp4', self.payload(out)[0]['md'])

    def test_inline_code_examples_are_not_packaged(self):
        text = '# Syntax\n\nUse `<video src="missing.mp4">` or ``![Example](missing.svg)`` as examples.\n'
        (self.docs / 'index.md').write_text(text)
        out = self.root / 'code.html'
        result = self.run_script('render_html.py', self.docs, '-o', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.payload(out)[0]['md'], text)

    def test_trace_produces_matching_svg_snapshots_and_safe_html(self):
        trace = json.loads((ROOT / 'assets' / 'prefix-cache-trace.json').read_text())
        trace['frames'][0]['title'] = 'Literal </script> & <b>title</b>'
        file = self.root / 'trace.json'; file.write_text(json.dumps(trace))
        out = self.root / 'trace.html'; frames = self.root / 'frames'
        result = self.run_script('render_trace.py', file, '-o', out, '--frames-dir', frames)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(re.search(r'<script id="trace-data" type="application/json">(.*?)</script>', out.read_text(), re.S).group(1))
        self.assertEqual(len(data['frames']), 3)
        for i, frame in enumerate(data['frames']):
            self.assertEqual(frame['svg'], (frames / f'frame-{i:03d}.svg').read_text())
        self.assertEqual((frames / 'poster.svg').read_text(), data['frames'][0]['svg'])
        self.assertIn('&lt;/script&gt;', data['frames'][0]['svg'])
        self.assertNotIn('<script src=', out.read_text())

    def test_trace_rejects_invalid_timing(self):
        file = self.root / 'invalid.json'
        file.write_text(json.dumps({'title': 'Bad', 'evidence_status': 'illustrative', 'frames': [{'title': 'State', 'duration': -1, 'panels': [{'label': 'A'}]}]}))
        result = self.run_script('render_trace.py', file, '-o', self.root / 'bad.html')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duration', result.stderr)

    def evidence_fixture(self):
        work = self.docs / '.work'; work.mkdir()
        (self.docs / 'kv.md').write_text('# Cache evidence\n')
        facts = {'steps': [{'id': 'kv.A1', 'input': [1, 2], 'output': 'miss', 'gap': 'No fixture', 'tests': []}], 'unverified': ['timing']}
        (work / 'kv.facts.json').write_text(json.dumps(facts))
        example = self.root / 'example.json'
        example.write_text(json.dumps({'evidence_status': 'illustrative', 'values': [1, 2], 'config': {'block_size': 4}, 'assumptions': ['empty cache']}))
        return work, example

    def test_evidence_survives_work_cleanup_with_uncertainty(self):
        work, example = self.evidence_fixture()
        result = self.run_script('collect_evidence.py', self.docs, '--commit', 'abcdef123456', '--example', example)
        self.assertEqual(result.returncode, 0, result.stderr)
        shutil.rmtree(work)
        ledger = json.loads((self.docs / 'evidence.json').read_text())
        self.assertEqual(ledger['source']['commit'], 'abcdef123456')
        self.assertEqual(ledger['subsystems'][0]['facts']['unverified'], ['timing'])
        self.assertEqual(ledger['subsystems'][0]['facts']['steps'][0]['gap'], 'No fixture')

    def test_duplicate_ids_preserve_existing_evidence(self):
        work, example = self.evidence_fixture()
        existing = self.docs / 'evidence.json'; existing.write_text('{"retained": true}')
        duplicate = json.loads((work / 'kv.facts.json').read_text())
        duplicate['doc'] = 'kv.md'
        (work / 'duplicate.facts.json').write_text(json.dumps(duplicate))
        result = self.run_script('collect_evidence.py', self.docs, '--commit', 'abcdef123456', '--example', example)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duplicate step ID', result.stderr)
        self.assertEqual(json.loads(existing.read_text()), {'retained': True})

    def test_evidence_document_mapping_is_explicit(self):
        work, example = self.evidence_fixture()
        (self.docs/'kv.md').rename(self.docs/'kv-cache.md')
        result = self.run_script('collect_evidence.py', self.docs, '--commit', 'abcdef123456', '--example', example)
        self.assertNotEqual(result.returncode,0)
        facts=json.loads((work/'kv.facts.json').read_text()); facts['doc']='kv-cache.md'
        (work/'kv.facts.json').write_text(json.dumps(facts))
        result = self.run_script('collect_evidence.py', self.docs, '--commit', 'abcdef123456', '--example', example)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads((self.docs/'evidence.json').read_text())['subsystems'][0]['doc'],'kv-cache.md')


if __name__ == '__main__':
    unittest.main()
