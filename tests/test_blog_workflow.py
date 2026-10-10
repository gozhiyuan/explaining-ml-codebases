import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
BLOG=ROOT/'skills/writing-ml-codebase-blogs'
spec=importlib.util.spec_from_file_location('animation_math',ROOT/'scripts/animation_math.py')
animation=importlib.util.module_from_spec(spec); spec.loader.exec_module(animation)


class BlogWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.article=self.root/'article.md'
        self.article.write_text('# A worked article\n\nQuestion and argument.\n')

    def tearDown(self): self.temp.cleanup()

    def run_script(self,file,*args,success=True):
        result=subprocess.run([sys.executable,str(file),*map(str,args)],capture_output=True,text=True)
        self.assertEqual(result.returncode==0,success,result.stdout+result.stderr)
        return result

    def test_attention_math_causal_and_unmasked(self):
        q=[[1,0],[0,1]]; v=[[1,0],[0,2]]
        causal=animation.attention(q,q,v)
        self.assertEqual(causal['weights'][0],[1,0])
        self.assertEqual(causal['output'][0],[1,0])
        expected=1/(1+math.exp(1/math.sqrt(2)))
        self.assertAlmostEqual(causal['weights'][1][0],expected)
        self.assertAlmostEqual(causal['output'][1][1],2*(1-expected))
        plain=animation.attention(q,q,v,False)
        self.assertGreater(plain['weights'][0][1],0)
        for row in plain['weights']: self.assertAlmostEqual(sum(row),1)
        with self.assertRaises(ValueError): animation.attention([[float('nan')]],[[0]],[[0]])
        with self.assertRaises(ValueError): animation.attention([[1,2]],[[1]],[[1]])

    def test_article_is_single_entry_and_bundles_measurement(self):
        (self.root/'unrelated.md').write_text('# Unrelated\nDO_NOT_INCLUDE_THIS_DOC')
        (self.root/'measurement.json').write_text('{"loss":2.0}')
        self.article.write_text('# Article\n[Measured result](measurement.json)\n')
        output=self.root/'export/article.html'
        self.run_script(BLOG/'scripts/render_article.py',self.article,'-o',output)
        page=output.read_text()
        self.assertIn('<body class="article">',page)
        self.assertNotIn('DO_NOT_INCLUDE_THIS_DOC',page)
        data=list(output.with_suffix('.assets').glob('measurement*.json'))
        self.assertEqual(len(data),1)
        self.assertEqual(json.loads(data[0].read_text()),{'loss':2.0})

    def test_jekyll_multiline_media_and_liquid_code(self):
        (self.root/'figure.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        (self.root/'clip.mp4').write_bytes(b'fixture')
        self.article.write_text('''# A worked article

![Example](figure.svg)
<video
 controls poster="figure.svg">
 <source src="clip.mp4" type="video/mp4">
</video>

$$
x = \\frac{1}{2}
$$

`{{ dangerous_inline }}`

```python
print("{{ dangerous_fenced }}")
```
''')
        stage=self.root/'stage'
        self.run_script(BLOG/'scripts/format_jekyll.py',self.article,'--profile',BLOG/'profiles/jekyll.json','--slug','worked-example','--date','2026-10-09','--out',stage)
        post=(stage/'_posts/2026-10-09-worked-example.md').read_text()
        self.assertNotIn('\n# A worked article',post)
        self.assertIn('relative_url',post)
        self.assertIn('assets/videos/worked-example/',post)
        self.assertIn('{% raw %}`{{ dangerous_inline }}`{% endraw %}',post)
        self.assertIn('{% raw %}\n```python',post)
        self.assertIn('<div class="math-block">',post)
        self.assertIn(r'\frac{1}{2}',post)
        manifest=json.loads((stage/'asset-manifest.json').read_text())
        for path in manifest['assets']: self.assertTrue((stage/path.lstrip('/')).is_file())
        before=(stage/'asset-manifest.json').read_bytes()
        self.run_script(BLOG/'scripts/format_jekyll.py',self.article,'--profile',BLOG/'profiles/jekyll.json','--slug','worked-example','--date','2026-10-09','--out',stage,success=False)
        self.assertEqual((stage/'asset-manifest.json').read_bytes(),before)

    def test_jekyll_failure_does_not_stage_partial_output(self):
        for text in ['# Title\n![missing](absent.svg)','[supporting docs](../index.md)']:
            self.article.write_text(text)
            stage=self.root/'failed-stage'
            self.run_script(BLOG/'scripts/format_jekyll.py',self.article,'--profile',BLOG/'profiles/jekyll.json','--slug','failure','--date','2026-10-09','--out',stage,success=False)
            self.assertFalse(stage.exists())

    def test_jekyll_restores_html_only_player_and_packages_it(self):
        (self.root/'player.html').write_text('<html><button>Next</button></html>')
        self.article.write_text('# Trace\n\n<!-- explain-html-only\n'
                               '<iframe src="player.html" title="Trace"></iframe>\n-->\n')
        stage=self.root/'stage'
        self.run_script(BLOG/'scripts/format_jekyll.py',self.article,
                        '--profile',BLOG/'profiles/jekyll.json','--slug','trace',
                        '--date','2026-10-09','--out',stage)
        post=(stage/'_posts/2026-10-09-trace.md').read_text()
        self.assertNotIn('explain-html-only',post)
        self.assertIn('<iframe src="{{ ',post)
        manifest=json.loads((stage/'asset-manifest.json').read_text())
        self.assertEqual(len(manifest['assets']),1)
        for path in manifest['assets']:
            self.assertEqual((stage/path.lstrip('/')).read_text(),'<html><button>Next</button></html>')

    def test_standalone_installed_blog_and_release_archive(self):
        dest=self.root/'installed'
        self.run_script(ROOT/'scripts/install_skills.py','--dest',dest,'--skill','blog')
        installed=dest/'writing-ml-codebase-blogs'
        self.assertTrue((installed/'scripts/manim_scenes.py').is_file())
        self.assertTrue((installed/'assets/manim-attention.json').is_file())
        self.run_script(installed/'scripts/render_article.py',self.article,'-o',self.root/'portable.html')
        before=(installed/'SKILL.md').read_bytes()
        self.run_script(ROOT/'scripts/install_skills.py','--dest',dest,'--skill','blog',success=False)
        self.assertEqual((installed/'SKILL.md').read_bytes(),before)
        archive=self.root/'release.zip'
        self.run_script(ROOT/'scripts/install_skills.py','--archive',archive,'--skill','all')
        import zipfile
        with zipfile.ZipFile(archive) as z:
            self.assertIn('explaining-ml-codebases/SKILL.md',z.namelist())
            self.assertIn('writing-ml-codebase-blogs/scripts/render_html.py',z.namelist())
            self.assertFalse(any('__pycache__' in name for name in z.namelist()))

    def test_claim_checker_requires_commit_and_retains_uncertainty(self):
        repo=self.root/'repo'; repo.mkdir()
        (repo/'model.py').write_text('def forward(x):\n    return x\n')
        for command in [['git','init','-q',str(repo)],['git','-C',str(repo),'add','.'],['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture']]:
            subprocess.run(command,check=True)
        sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        data={'title':'Article','question':'How?','audience':'Engineers','scope':'Forward','source':{'commit':sha},'claims':[{'id':'one','evidence_status':'source-reviewed','sources':[{'file':'model.py','symbol':'forward'}]}],'uncertainties':[]}
        manifest=self.root/'article.json'; manifest.write_text(json.dumps(data))
        checker=BLOG/'scripts/check_article.py'
        self.run_script(checker,self.article,'--manifest',manifest,'--repo',repo)
        evidence=self.root/'evidence.json'; evidence.write_text(json.dumps({'source':{'commit':sha},'subsystems':[{'facts':{'unverified':['GPU was not run']}}]}))
        self.run_script(checker,self.article,'--manifest',manifest,'--repo',repo,'--evidence',evidence,success=False)
        data['uncertainties']=['GPU was not run']; data['claims'][0]['sources'][0]['symbol']='<module>'
        manifest.write_text(json.dumps(data))
        self.run_script(checker,self.article,'--manifest',manifest,'--repo',repo,'--evidence',evidence)
        data['source']['commit']='0000000'; manifest.write_text(json.dumps(data))
        self.run_script(checker,self.article,'--manifest',manifest,'--repo',repo,success=False)

    def test_manim_missing_backend_does_not_claim_success(self):
        output=self.root/'unrendered.mp4'
        self.run_script(ROOT/'scripts/render_manim.py',ROOT/'assets/manim-attention.json','-o',output,'--python',self.root/'no-python',success=False)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.metadata.json').exists())

    def test_python_config_discovery_does_not_label_model_code(self):
        config=self.root/'config'; config.mkdir()
        (config/'train_small.py').write_text('batch_size = 2\n')
        (self.root/'model.py').write_text('class Config:\n    pass\n')
        result=self.run_script(ROOT/'scripts/scan_ml_repo.py',self.root,'--json')
        configs=json.loads(result.stdout)['configs']
        self.assertIn('config/train_small.py',configs)
        self.assertNotIn('model.py',configs)


if __name__=='__main__': unittest.main()
