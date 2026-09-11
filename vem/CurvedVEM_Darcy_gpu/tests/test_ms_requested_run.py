"""临时验收：独立复算本次 MS 六层实验的尺度、收敛阶、终点和残差。"""
import csv
import math
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / 'output/MS_triblock_bdf2_k1/convergence.csv').open()))
assert len(rows) == 18, f'需要六层三组分，实际 {len(rows)} 行'
previous = {}
for row in rows:
    side = int(row['mesh'].split('x')[0])
    c = int(row['component'])
    folder = root / 'output/MS_triblock_bdf2_k1' / ('mesh_' + row['mesh'])
    report = json.loads((folder / 'report.json').read_text())
    assert report['status'] == 'converged' and report['k'] == 1 and report['example'] == 4
    assert not (folder / 'solution.csv').exists()
    # 每次 Picard 内的线性真实残差和最终非线性残差分别验收。
    for line in (folder / 'iterations.log').read_text().splitlines():
        if line.startswith('step=') and ' relative_change=' in line:
            values = dict(item.split('=') for item in line.split())
            assert float(values['residual']) <= 1e-11 * (1 + 1e-10)
    assert side in (4, 8, 16, 32, 64, 128) and c in (0, 1, 2)
    assert int(row['cells']) == side * side
    h, dt, t = (float(row[key]) for key in ('h', 'dt', 'time'))
    assert math.isclose(h, 1 / side, rel_tol=1e-9)
    assert math.isclose(dt * int(row['steps']), 0.1, abs_tol=1e-14)
    assert math.isclose(t, 0.1, abs_tol=1e-14)
    assert dt <= 0.1 * h * (1 + 1e-10)
    assert float(row['residual']) <= 1e-6
    assert float(row['total_seconds']) >= float(row['iteration_seconds']) > 0
    for field in ('scalar', 'flux'):
        err, order = (float(row[f'{field}_{key}']) for key in ('error', 'order'))
        assert math.isfinite(err) and err > 0
        if c in previous:
            old = previous[c]
            expected = math.log(float(old[f'{field}_error']) / err) / math.log(float(old['h']) / h)
            assert math.isclose(order, expected, rel_tol=1e-10, abs_tol=1e-10)
        else:
            assert math.isnan(order)
    previous[c] = row
    print(f'PASS {row["mesh"]} component={c} T={t} dt={dt} residual={row["residual"]}')
assert {r['mesh'] for r in rows} == {f'{n}x{n}' for n in (4, 8, 16, 32, 64, 128)}
print('PASS: 六层三组分的终点、尺度、误差阶、残差和耗时全部通过独立核查。')
