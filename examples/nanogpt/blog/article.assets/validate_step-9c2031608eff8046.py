"""Read-only original nanoGPT model + AST-extracted batch/LR functions, CPU fixture."""
import argparse, ast, importlib.util, json, math, platform, tempfile, sys, subprocess, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
torch.set_num_threads(1)
repo = args.repo.resolve()
commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
assert commit == '3adf61e154c3fe3fca428ad6bc3818b27a3b8291', 'use the pinned evaluation checkout'
spec = importlib.util.spec_from_file_location('nanogpt_model', repo / 'model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = ast.parse((repo / 'train.py').read_text())
functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in ('get_batch', 'get_lr')]
assert len(functions) == 2
stream = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
config = dict(block_size=4, vocab_size=8, n_layer=1, n_head=2, n_embd=8, dropout=0.0, bias=False)
with tempfile.TemporaryDirectory(dir=args.output.parent) as fixture_dir:
    np.array(stream, dtype=np.uint16).tofile(Path(fixture_dir) / 'train.bin')
    namespace = dict(np=np, torch=torch, os=__import__('os'), math=math, data_dir=fixture_dir,
                     block_size=4, batch_size=2, device_type='cpu', device='cpu',
                     learning_rate=0.004, warmup_iters=3, lr_decay_iters=11, min_lr=0.0004)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(repo / 'train.py'), 'exec'), namespace)
    torch.manual_seed(1337)
    x, y = namespace['get_batch']('train')
    assert x.shape == y.shape == (2, 4) and x.dtype == y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.tolist() == [[7,0,1,2],[5,6,7,0]] and y.tolist() == [[0,1,2,3],[6,7,0,1]]
    torch.manual_seed(1337)
    model = module.GPT(module.GPTConfig(**config)).train()
    assert model.transformer.wte.weight is model.lm_head.weight
    shapes = {}
    for name in ['transformer.wte', 'transformer.wpe', 'transformer.h.0', 'transformer.h.0.ln_1', 'transformer.h.0.attn', 'transformer.h.0.mlp.c_fc', 'transformer.h.0.mlp', 'transformer.ln_f', 'lm_head']:
        model.get_submodule(name).register_forward_hook(lambda m, a, out, n=name: shapes.__setitem__(n, list(out.shape)))
    logits, loss = model(x, y)
    before_shapes = dict(shapes)
    with torch.no_grad():
        embedded = model.transformer.wte(x) + model.transformer.wpe(torch.arange(4))
        normed = model.transformer.h[0].ln_1(embedded)
        q, k, v = model.transformer.h[0].attn.c_attn(normed).split(8, dim=2)
        q = q.view(2, 4, 2, 4).transpose(1, 2)
        k = k.view(2, 4, 2, 4).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
        scores = scores.masked_fill(torch.tril(torch.ones(4,4)) == 0, -float('inf'))
        attention = scores.softmax(-1)
        first_hidden = model.transformer.ln_f(model.transformer.h[0](embedded))
    assert logits.shape == (2, 4, 8) and loss.ndim == 0 and torch.isfinite(loss)
    assert math.isclose(loss.item(), 2.0568580627441406, abs_tol=1e-5), 'article values no longer match this runtime'
    per_position = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(per_position.mean(), loss)
    # Causal invariance: replacing the last input must not change earlier logits.
    modified = x.clone(); modified[:, -1] = (modified[:, -1] + 1) % 8
    with torch.no_grad():
        changed_logits, _ = model(modified, y)
        inference_logits, inference_loss = model(x)
    torch.testing.assert_close(changed_logits[:, :-1], logits[:, :-1], rtol=1e-5, atol=1e-6)
    assert inference_logits.shape == (2, 1, 8) and inference_loss is None
    lr = namespace['get_lr'](0)
    assert lr == 0.001
    optimizer = model.configure_optimizers(0.1, lr, (0.9, 0.95), 'cpu')
    assert [g['weight_decay'] for g in optimizer.param_groups] == [0.1, 0.0]
    assert all(p.dim() >= 2 for p in optimizer.param_groups[0]['params'])
    assert all(p.dim() < 2 for p in optimizer.param_groups[1]['params'])
    initial = model.transformer.wte.weight.detach().clone()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    grad_sample = model.transformer.wte.weight.grad[0, :4].tolist()
    norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    norm_after = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert norm_after <= 1.00001
    optimizer.step()
    delta = (model.transformer.wte.weight.detach() - initial).abs().max().item()
    assert delta > 0
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        _, after_loss = model(x, y)
    assert after_loss < loss
    assert math.isclose(after_loss.item(), 2.0441248416900635, abs_tol=1e-5), 'article values no longer match this runtime'
    assert sum(p.numel() for p in model.parameters()) == 888
    result = dict(evidence_status='measured', source_commit=commit,
        source_sha256={name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in ('train.py','model.py')},
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         platform=platform.system(), machine=platform.machine(), device='cpu', threads=1),
        seed=1337, config=config, fixture_stream=stream, x=x.tolist(), y=y.tolist(), shapes=before_shapes,
        logits_first=logits[0, 0].tolist(), probabilities_first=logits[0, 0].softmax(-1).tolist(),
        per_position_loss=per_position.tolist(), loss_before=loss.item(), loss_after=after_loss.item(),
        embedding_first=embedded[0,0].tolist(), normalized_first=normed[0,0].tolist(),
        attention_head0=attention[0,0].tolist(), final_hidden_first=first_hidden[0,0].tolist(),
        lr_schedule={str(i): namespace['get_lr'](i) for i in (0, 2, 3, 7, 11, 12)},
        lr=lr, grad_sample=grad_sample, grad_norm_before=norm_before.item(), grad_norm_after=norm_after.item(),
        max_embedding_change=delta, parameter_count=sum(p.numel() for p in model.parameters()),
        uses_sdpa=model.transformer.h[0].attn.flash,
        checks=['original get_batch int64 shifted windows','weight sharing identity','forward shapes',
                'explicit negative-log-probability equals original cross entropy','causal prefix invariance',
                'last-position inference shape','source warmup LR','AdamW parameter groups',
                'finite gradients','clip bound','parameter update','gradients cleared','same-batch loss falls'])
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
print('PASS: 13 assertion groups, one tiny CPU update. Full train.py process not executed.')
