"""Dependency-free validation and numerical contract for original scenes."""
import math


def attention(q, k, v, causal=True):
    def matrix(value, name):
        if not isinstance(value, list) or not value or not all(isinstance(r, list) and r for r in value):
            raise ValueError(f'{name} must be a nonempty rectangular matrix')
        width = len(value[0])
        if any(len(r) != width for r in value):
            raise ValueError(f'{name} must be rectangular')
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for r in value for x in r):
            raise ValueError(f'{name} must contain finite numbers')
        return width
    d = matrix(q, 'q')
    if matrix(k, 'k') != d:
        raise ValueError('q and k feature dimensions must agree')
    matrix(v, 'v')
    if len(k) != len(v) or len(q) != len(k):
        raise ValueError('this self-attention scene requires equal sequence lengths')
    if len(q) > 4 or d > 4 or len(v[0]) > 4:
        raise ValueError('use at most 4 rows and 4 features for a readable scene')
    scores = [[sum(a*b for a,b in zip(row, key))/math.sqrt(d) for key in k] for row in q]
    masked = [[s if not causal or j <= i else None for j,s in enumerate(row)] for i,row in enumerate(scores)]
    weights = []
    for row in masked:
        peak = max(s for s in row if s is not None)
        exps = [math.exp(s-peak) if s is not None else 0.0 for s in row]
        total = sum(exps)
        weights.append([x/total for x in exps])
    out = [[sum(w*v[j][c] for j,w in enumerate(row)) for c in range(len(v[0]))] for row in weights]
    return {'scores': scores, 'masked_scores': masked, 'weights': weights, 'output': out}


def validate(spec):
    template = spec.get('template')
    if template not in {'attention', 'token-flow', 'cache-reuse'}:
        raise ValueError('template must be attention, token-flow or cache-reuse')
    if not isinstance(spec.get('title'), str) or not spec['title'].strip():
        raise ValueError('title is required')
    if template == 'attention':
        if not isinstance(spec.get('causal', True), bool):
            raise ValueError('causal must be boolean')
        return attention(spec['q'], spec['k'], spec['v'], spec.get('causal', True))
    if template == 'token-flow':
        stages = spec.get('stages', [])
        if not 2 <= len(stages) <= 8:
            raise ValueError('token-flow requires 2–8 stages')
        for stage in stages:
            if not all(isinstance(stage.get(s), str) and stage[s] for s in ('label', 'shape', 'detail')):
                raise ValueError('each stage requires label, shape and detail strings')
        return {'stages': stages}
    blocks = spec.get('blocks', [])
    if not 1 <= len(blocks) <= 3:
        raise ValueError('cache-reuse requires 1–3 blocks')
    for block in blocks:
        if not all(isinstance(block.get(s), str) and block[s] for s in ('tokens', 'hash', 'page')):
            raise ValueError('each block requires tokens, hash and page strings')
    return {'blocks': blocks, 'hashes_are_illustrative': True}
