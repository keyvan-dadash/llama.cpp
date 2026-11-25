#!/usr/bin/env python3
"""Four-way K3 benchmarks: existing binaries only, no Git or CMake.

python3 k3_bench.py                      # 10 sizes, prompt + decode/history
python3 k3_bench.py --suite all          # add incremental prefill + generation
python3 k3_bench.py --suite generation   # increasing generation, d64
python3 k3_bench.py --suite legacy       # reproduce old A100 sweep exactly
python3 k3_bench.py --plot-only          # last saved plan; no binaries needed

Each repetition is a fresh llama-bench -r 1 process. Automatically imports
compatible earlier CSV results and resumes missing points. No source changes.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

HOME = Path.home() / 'keyvan'
REPO = HOME / 'source-llama/llama.cpp'
OLD = HOME / 'a100-benchmark-results'
FOUR = HOME / 'k3-four-way-results'
SIZES = [64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
GEN = [16, 32, 64, 128, 256, 512, 1024, 2048]
VARIANTS = {
    'baseline': ('A100 original repack', '8-11', '0', 'o', '#1f77b4'),
    'repack': ('A100 optimized repack', '8-11', '0', 's', '#ff7f0e'),
    'fusion': ('A100 repack + fusion', '8-11', '1', '^', '#2ca02c'),
    'x100': ('X100 repack', '0-3', '0', 'D', '#d62728'),
}
ALIASES = {'a100_repack': 'repack', 'a100_fusion': 'fusion'}
FIELDS = ('variant', 'prompt', 'generation', 'depth', 'repeat', 'tps', 'avg_ns',
          'elapsed_s', 'build_commit', 'model_size', 'model_n_params', 'log', 'source')
SETTINGS = {'threads': 4, 'ubatch': 64, 'batch': 'max(64, prompt)',
            'kv': 'f16', 'repack': 1, 'flash_attn': 1, 'ngl': 0,
            'no_warmup': True, 'llama_bench_reps': 1}


def path(p):
    return Path(p).expanduser().resolve()


def read_json(p):
    return json.loads(p.read_text())


def write_json(p, data):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(p)


def read_rows(p):
    if not p.is_file():
        return []
    with p.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r['variant'] not in VARIANTS and r['variant'] not in ALIASES:
            raise ValueError(f'Unknown variant in {p}: {r["variant"]}')
        if not math.isfinite(float(r['tps'])) or float(r['tps']) <= 0:
            raise ValueError(f'Invalid throughput in {p}')
    return rows


def key(r):
    return (ALIASES.get(r['variant'], r['variant']),
            *(int(r[k]) for k in ('prompt', 'generation', 'depth', 'repeat')))


def save_rows(p, rows):
    tmp = p.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(rows.values(), key=key))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(p):
    s = p.stat()
    return [str(p.resolve()), s.st_size, s.st_mtime_ns]


def artifacts(binary):
    files = [binary, *sorted(binary.parent.glob('lib*.so*'))]
    memo = {}
    result = {}
    for f in files:
        if f.is_file():
            real = f.resolve()
            if real not in memo:
                memo[real] = sha(real)
            result[f.name] = memo[real]
    return result


def named_hashes(mapping):
    return {Path(k).name: v for k, v in mapping.items()}


def make_plan(a):
    if a.suite == 'legacy':
        return read_json(a.source / 'manifest.json')['plan']
    sizes = sorted(set(a.sizes))
    depths = sorted(set(a.depths if a.depths is not None else sizes))
    gen = sorted(set(a.generation_sizes))
    def sweep(title, xlabel, xs, fn, phase):
        return {'title': title, 'xlabel': xlabel, 'x': xs,
                'cases': [fn(x) for x in xs], 'phase': phase}
    pp = sweep('Full prompt (empty cache)', 'Prompt tokens', sizes,
               lambda s: (s, 0, 0), 'prefill')
    ph = sweep('64-token prefill vs. history', 'Initial KV tokens', depths,
               lambda s: (64, 0, s), 'prefill')
    dh = sweep(f'{a.decode_tokens}-token decode vs. history', 'Initial KV tokens', depths,
               lambda s: (0, a.decode_tokens, s), 'decode')
    dg = sweep(f'Generation length (initial depth {a.initial_depth})', 'Generated tokens', gen,
               lambda s: (0, s, a.initial_depth), 'decode')
    if a.suite == 'generation':
        sweeps = [dg]
    elif a.suite == 'all':
        sweeps = [pp, ph, dg, dh]
    else:
        sweeps = [pp if a.prefill_mode == 'prompt' else ph, dh]
    cases = list(dict.fromkeys(tuple(c) for s in sweeps for c in s['cases']))
    return {'sweeps': sweeps, 'anchors': [], 'cases': cases}


def resolve_binaries(a):
    # The baseline built by the earlier script is reused, never compiled here.
    baseline = a.baseline_binary
    if baseline is None and (a.source / 'manifest.json').is_file():
        baseline = read_json(a.source / 'manifest.json').get('binaries', {}).get('baseline')
    baseline = baseline or REPO.parent / 'a100-baseline-6669fc09a/baseline-build/bin/llama-bench'
    current = path(a.binary)
    base = path(baseline)
    if base == current:
        raise ValueError('Original A100 baseline needs a distinct executable and shared libraries')
    for b in (base, current):
        if not b.is_file() or not os.access(b, os.X_OK):
            raise ValueError(f'Missing existing executable: {b}\nUse --baseline-binary or --binary. No build is performed.')
    return {'baseline': base, 'repack': current, 'fusion': current, 'x100': current}


def make_identity(bins, model):
    base_hash = artifacts(bins['baseline'])
    cur_hash = artifacts(bins['repack'])
    if base_hash == cur_hash:
        raise ValueError('Baseline and optimized build artifacts are identical')
    return {'schema': 'k3-four-way-v1', 'settings': SETTINGS,
            'variants': {k: list(v[:3]) for k, v in VARIANTS.items()},
            'model': fingerprint(model),
            'binaries': {k: str(b) for k, b in bins.items()},
            'artifacts': {'baseline': base_hash, 'repack': cur_hash,
                          'fusion': cur_hash, 'x100': cur_hash}}


def compatible_legacy(m, identity, variant):
    model = identity['model']
    # Old sweep recorded size + nanosecond mtime, not a full model hash.
    if (int(m['model_size']), int(m['model_mtime_ns'])) != (model[1], model[2]):
        return False
    old_variant = 'repack' if variant == 'x100' else variant
    return named_hashes(m['artifacts'][old_variant]) == identity['artifacts'][variant]


def import_results(a, out, identity, bins, rows):
    """Import only matching provenance. Earlier source directories are read-only."""
    if a.no_reuse:
        return
    dirs = [a.source, FOUR, REPO / 'k3-scaling-results', Path.cwd() / 'k3-scaling-results', *a.reuse]
    for src in dict.fromkeys(path(s) for s in dirs):
        if src == out or not (src / 'manifest.json').is_file() or not (src / 'results.csv').is_file():
            continue
        added = 0
        try:
            m = read_json(src / 'manifest.json')
            if 'source_manifest_sha256' in m:
                # The old add-X100 script ties its data to the original A100 manifest and CSV.
                parent = path(m['source'])
                if (sha(parent / 'manifest.json') != m['source_manifest_sha256'] or
                        sha(parent / 'results.csv') != m['source_results_sha256'] or
                        m.get('x100_cores') != '0-3' or int(m.get('graph', -1)) != 0):
                    raise ValueError('original A100 source changed')
                old = read_json(parent / 'manifest.json')
                eligible = {v for v in VARIANTS if compatible_legacy(old, identity, v)}
            elif 'head' in m and 'artifacts' in m:
                eligible = {v for v in ('baseline', 'repack', 'fusion')
                            if compatible_legacy(m, identity, v)}
            elif m.get('schema') == 'k3-four-way-v1':
                eligible = {v for v in VARIANTS if m['model'][1:] == identity['model'][1:]
                            and m['settings'] == SETTINGS and m['variants'][v] == list(VARIANTS[v][:3])
                            and m['artifacts'][v] == identity['artifacts'][v]}
            elif 'binary' in m and 'libraries' in m:
                # k3_scaling used path/size/mtime fingerprints rather than SHA256.
                current = bins['repack']
                libs = sorted({f.resolve() for f in current.parent.glob('lib*.so*') if f.is_file()})
                same = (m['binary'] == fingerprint(current) and m['model'] == identity['model']
                        and m['libraries'] == [fingerprint(f) for f in libs]
                        and m['batch'] == SETTINGS['batch'] and m['ubatch'] == 64
                        and m['threads'] == 4 and m['kv'] == 'f16' and m['repack'] == 1)
                eligible = {ALIASES.get(v, v) for v, spec in m['variants'].items()
                            if ALIASES.get(v, v) in ('repack', 'fusion', 'x100')
                            and spec[1:] == list(VARIANTS[ALIASES.get(v, v)][1:3])} if same else set()
            else:
                raise ValueError('unrecognized manifest')
            for oldrow in read_rows(src / 'results.csv'):
                k = key(oldrow)
                if k[0] not in eligible or k in rows:
                    continue
                r = dict(oldrow, variant=k[0], log=str((src / oldrow['log']).resolve()), source=str(src))
                rows[k] = r
                added += 1
            print(f'Imported {added} matching points from {src}; eligible: {", ".join(sorted(eligible)) or "none"}', flush=True)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            print(f'Skipped import from {src}: {exc}', flush=True)


def make_command(binary, model, variant, case):
    p, n, d = case
    _, cores, graph, _, _ = VARIANTS[variant]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('LLAMA_RVV1024_', 'GGML_RVV')) and k not in
           ('LLAMA_KV_ROWS_DISABLE', 'GGML_BACKEND_PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH')}
    env.update(LLAMA_RVV1024_GRAPH=graph, LD_LIBRARY_PATH=str(binary.parent))
    cmd = ['taskset', '-c', cores, str(binary), '-m', str(model),
           '-p', str(p), '-n', str(n), '-d', str(d), '-r', '1',
           '-b', str(max(64, p)), '-ub', '64', '-t', '4', '-ngl', '0',
           '-ctk', 'f16', '-ctv', 'f16', '-fa', 'on', '--repack', '1', '--no-warmup', '-o', 'jsonl']
    if cores == '8-11':
        cmd = ['bash', '-c', 'printf "%s\\n" "$BASHPID" > /proc/set_ai_thread || exit 1; exec "$@"',
               'a100-bench', *cmd]
    return cmd, env


def parse_result(stdout, stderr, variant, case):
    records = [json.loads(line) for line in stdout.splitlines() if line.lstrip().startswith('{')]
    if len(records) != 1:
        raise RuntimeError(f'Expected one JSONL result, got {len(records)}')
    r = records[0]
    p, n, d = case
    expected = dict(n_prompt=p, n_gen=n, n_depth=d, n_batch=max(64, p), n_ubatch=64,
                    n_threads=4, n_gpu_layers=0, flash_attn=1)
    for k, v in expected.items():
        if int(r[k]) != v:
            raise RuntimeError(f'Unexpected {k}: {r[k]}, expected {v}')
    if r['repack'] is not True or r['type_k'] != 'f16' or r['type_v'] != 'f16':
        raise RuntimeError('Expected repack=1 and FP16 KV')
    logs = stdout + '\n' + stderr
    q6 = 'Q6_K 64x1 repack:' in logs
    pp = 'BQ8 KCOL-QK:' in logs
    tg = 'decode6-split4:' in logs
    if q6 != (variant in ('repack', 'fusion')):
        raise RuntimeError('Q6_K repack banner does not match requested variant')
    if variant == 'fusion':
        if (p >= 64 or d >= 64) and not pp or n > 0 and not tg:
            raise RuntimeError('Expected fused kernel banner missing')
    elif pp or tg:
        raise RuntimeError('Unexpected fusion with GRAPH=0')
    tps = float(r['avg_ts'])
    if not math.isfinite(tps) or tps <= 0:
        raise RuntimeError('Invalid throughput')
    return r, tps


def run_case(binary, model, variant, case, repeat, out):
    p, n, d = case
    stem = out / 'logs' / f'{variant}_p{p}_n{n}_d{d}_r{repeat}'
    cmd, env = make_command(binary, model, variant, case)
    stem.with_suffix('.command.txt').write_text(
        'LLAMA_RVV1024_GRAPH=' + env['LLAMA_RVV1024_GRAPH'] + '\nLD_LIBRARY_PATH=' +
        env['LD_LIBRARY_PATH'] + '\n' + shlex.join(cmd) + '\n')
    start = time.monotonic()
    so, se = stem.with_suffix('.jsonl'), stem.with_suffix('.stderr.txt')
    with so.open('w') as stdout, se.open('w') as stderr:
        proc = subprocess.run(cmd, cwd=binary.parent, env=env, stdout=stdout, stderr=stderr)
    if proc.returncode:
        raise RuntimeError(f'exit {proc.returncode}; see {se}')
    r, tps = parse_result(so.read_text(), se.read_text(), variant, case)
    return dict(zip(FIELDS, (variant, p, n, d, repeat, tps, r['avg_ns'],
                             round(time.monotonic()-start, 3), r['build_commit'],
                             r['model_size'], r['model_n_params'], str(stem), 'measured')))


def plots(out, plan, rows, reps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = {}
    for k, r in rows.items():
        if k[-1] <= reps:
            groups.setdefault(k[:-1], []).append(float(r['tps']))
    means = {k: statistics.mean(v) for k, v in groups.items()}
    for phase in ('prefill', 'decode'):
        sweeps = [s for s in plan['sweeps'] if s.get('phase',
                  'prefill' if s['cases'][0][0] > 0 else 'decode') == phase]
        if not sweeps:
            continue
        fig, axes = plt.subplots(1, len(sweeps), figsize=(7.5*len(sweeps), 5.5),
                                 squeeze=False, constrained_layout=True)
        for ax, s in zip(axes[0], sweeps):
            xs = s['x']
            for v, (label, _, _, marker, color) in VARIANTS.items():
                ys = [means.get((v, *c), float('nan')) for c in s['cases']]
                ax.plot(xs, ys, marker=marker, color=color, linewidth=2, label=label)
            if all(x > 0 for x in xs):
                ax.set_xscale('log', base=2)
            else:
                ax.set_xscale('symlog', base=2, linthresh=64)
            ax.set_xticks(xs, [f'{x//1024}K' if x >= 1024 and x % 1024 == 0 else str(x) for x in xs], rotation=35)
            ax.set_title(s['title'])
            ax.set_xlabel(s['xlabel'])
            ax.set_ylabel('Tokens / second')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(fontsize=9)
        fig.suptitle(f'SpacemiT K3 | {phase.capitalize()} | 4 cores, ubatch 64, FP16 KV')
        fig.savefig(out / f'{phase}.png', dpi=180)
        plt.close(fig)
    if plan.get('anchors'):
        anchors = plan['anchors']
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for j, (v, (label, _, _, _, color)) in enumerate(VARIANTS.items()):
            ys = [means.get((v, *c), float('nan')) for c in anchors]
            ax.bar([i+(j-1.5)*.19 for i in range(len(anchors))], ys, .19, label=label, color=color)
        ax.set_xticks(range(len(anchors)), [f'pp{p} @ d{d}' for p, n, d in anchors])
        ax.set_ylabel('Tokens / second')
        ax.set_title('Reference prefill workloads')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=.25)
        fig.savefig(out / 'prefill_reference_cases.png', dpi=180)
        plt.close(fig)
    print(f'Plots and CSV: {out}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binary', type=Path, default=REPO/'build-rvv1024/bin/llama-bench')
    ap.add_argument('--baseline-binary', type=Path, help='Existing original-repack llama-bench (separate build)')
    ap.add_argument('--model', type=Path, default=HOME/'swe-k3/models/qwen2.5-coder-1.5b-instruct-q4_0-rvv1024.gguf')
    ap.add_argument('--source', type=path, default=OLD, help='Earlier A100 results directory')
    ap.add_argument('--output', type=path, default=HOME/'k3-unified-results')
    ap.add_argument('--reuse', type=path, action='append', default=[], help='Additional earlier result directories')
    ap.add_argument('--no-reuse', action='store_true', help='Do not import old result directories (still resumes this output)')
    ap.add_argument('--suite', choices=('context', 'all', 'generation', 'legacy'), default='context')
    ap.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    ap.add_argument('--depths', nargs='+', type=int, help='Override history depths (0 allowed)')
    ap.add_argument('--generation-sizes', nargs='+', type=int, default=GEN)
    ap.add_argument('--initial-depth', type=int, default=64)
    ap.add_argument('--decode-tokens', type=int, default=64)
    ap.add_argument('--prefill-mode', choices=('prompt', 'history'), default='prompt')
    ap.add_argument('--reps', type=int, choices=(1, 2), default=1, help='Fresh -r 1 processes per point')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--plot-only', action='store_true', help='Replot saved plan and CSV, no benchmark execution')
    a = ap.parse_args()
    out = path(a.output)
    if out in {path(a.source), path(FOUR), path(REPO/'k3-scaling-results'), path(Path.cwd()/'k3-scaling-results'), *[path(s) for s in a.reuse]}:
        ap.error('Choose a new --output directory to preserve the older results')
    if a.plot_only:
        last = read_json(out/'plan.json')
        rows = {key(r): r for r in read_rows(out/'results.csv')}
        if not rows:
            raise ValueError('No saved measurements to plot')
        plots(out, last['plan'], rows, last['reps'])
        return
    if (any(x <= 0 or x % 64 for x in a.sizes) or
            any(x < 0 or x % 64 for x in (a.depths or [])) or
            a.initial_depth < 0 or a.initial_depth % 64 or
            min(a.generation_sizes) <= 0 or a.decode_tokens <= 0):
        ap.error('Prompt sizes must be positive multiples of 64; history depths nonnegative multiples of 64; generation positive')
    plan = make_plan(a)
    cases = list(dict.fromkeys(tuple(map(int, c)) for c in plan['cases']))
    wanted = [(v, *c, r) for i, c in enumerate(cases) for r in range(1, a.reps+1)
              for v in list(VARIANTS)[(i+r-1)%4:] + list(VARIANTS)[:(i+r-1)%4]]
    print(f'{len(cases)} workloads x 4 configurations x {a.reps} repeat(s) = {len(wanted)} points before reuse', flush=True)
    if a.plan:
        for p, n, d in cases:
            print(f'  -p {p:5} -n {n:5} -d {d:5}')
        print('No benchmarks, builds or Git operations performed.')
        return
    import matplotlib  # check before any benchmark
    bins, model = resolve_binaries(a), path(a.model)
    if not model.is_file():
        raise ValueError(f'Model not found: {model}')
    identity = make_identity(bins, model)
    out.mkdir(parents=True, exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    csv_path, mp = out/'results.csv', out/'manifest.json'
    if mp.exists():
        if read_json(mp) != identity:
            raise ValueError('Build/model/settings changed. Use a new --output directory.')
    elif csv_path.exists():
        raise ValueError('CSV without manifest. Use a new --output directory.')
    else:
        write_json(mp, identity)
    rows = {key(r): r for r in read_rows(csv_path)}
    import_results(a, out, identity, bins, rows)
    save_rows(csv_path, rows)
    write_json(out/'plan.json', {'plan': plan, 'reps': a.reps})
    pending = [k for k in wanted if k not in rows]
    print(f'Reusing {len(wanted)-len(pending)} requested points; {len(pending)} new runs.', flush=True)
    print(f'Untimed history fill for pending runs: {sum(k[3] for k in pending):,} tokens', flush=True)
    if any(k[0] != 'x100' for k in pending) and not os.access('/proc/set_ai_thread', os.W_OK):
        raise ValueError('A100 registration unavailable: /proc/set_ai_thread')
    failures = []
    try:
        for i, (v, p, n, d, r) in enumerate(pending, 1):
            print(f'[{i}/{len(pending)}] {VARIANTS[v][0]}: p{p} n{n} d{d} repeat {r}', flush=True)
            try:
                row = run_case(bins[v], model, v, (p, n, d), r, out)
                # All variants must load the same model. Old scaling CSV may lack metadata.
                infos = {(int(x['model_size']), int(x['model_n_params'])) for x in rows.values()
                         if x.get('model_size') and x.get('model_n_params')}
                if infos and infos != {(int(row['model_size']), int(row['model_n_params']))}:
                    raise RuntimeError('Model size/parameter count differs from saved results')
                rows[key(row)] = row
                save_rows(csv_path, rows)
                print(f'  {row["tps"]:.2f} t/s ({row["elapsed_s"]:.1f}s total)', flush=True)
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                failures.append(f'{v} p{p} n{n} d{d} r{r}: {exc}')
                print(f'FAILED: {failures[-1]}', flush=True)
    except KeyboardInterrupt:
        print('Interrupted; rerun to resume missing points.', flush=True)
    finally:
        if rows:
            plots(out, plan, rows, a.reps)
        if failures:
            (out/'failures.txt').write_text('\n'.join(failures)+'\n')
        else:
            (out/'failures.txt').unlink(missing_ok=True)
    if failures:
        raise SystemExit(f'{len(failures)} failed points; see failures.txt; rerun to retry.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
