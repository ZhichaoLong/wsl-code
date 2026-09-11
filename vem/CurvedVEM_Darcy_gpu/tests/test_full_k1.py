"""逐层核对 k=1 正式运行结果；--final 还要求六层和监控进程全部完成。"""
import argparse,csv,json,math,re
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--final',action='store_true')
parser.add_argument('--root',type=Path,default=Path('output/MS_triblock_bdf2_k1'))
args=parser.parse_args()
root=args.root
rows=list(csv.DictReader((root/'convergence.csv').open()))
assert all(None not in row for row in rows)
labels=list(dict.fromkeys(r['mesh'] for r in rows))
if args.final:
 assert labels==['4x4','8x8','16x16','32x32','64x64','128x128']
 monitor=json.loads((root/'monitor/summary.json').read_text())
 assert monitor['returncode']==0 and monitor['stop_reason'] is None
previous={};linear=[]
for label in labels:
 m=int(label.split('x')[0]);folder=root/f'mesh_{label}'
 report=json.loads((folder/'report.json').read_text())
 assert report['status']=='converged' and report['k']==1 and report['example']==4
 assert report['dofs']==30*m*m+12*m and report['cells']==m*m
 assert abs(report['time']-.1)<1e-14
 assert report['petsc_ordering']=='nd'
 selected=[r for r in rows if r['mesh']==label]
 assert len(selected)==3
 for row in selected:
  assert int(row['steps'])==m and abs(float(row['dt'])*m-.1)<1e-14
  assert float(row['residual'])<=1e-6
  assert 0<float(row['ksp_seconds'])<=float(row['linear_solve_seconds'])<=float(row['iteration_seconds'])
  assert math.isclose(float(row['average_step_seconds'])*m,float(row['iteration_seconds']),rel_tol=1e-12)
  c=row['component']
  if c in previous:
   old=previous[c]
   for error,order in [('scalar_error','scalar_order'),('flux_error','flux_order')]:
    value=math.log(float(old[error])/float(row[error]))/math.log(float(old['h'])/float(row['h']))
    assert abs(value-float(row[order]))<1e-12
  previous[c]=row
 steps=[]
 for line in (folder/'iterations.log').read_text().splitlines():
  if re.match(r'step=\d+ step_seconds=',line):
   s=dict(field.split('=',1) for field in line.split());steps.append(s)
   assert int(s['rss_bytes'])>0 and int(s['gpu_used_bytes'])>0
  elif re.match(r'step=\d+ picard=',line):
   s=dict(field.split('=',1) for field in line.split())
   residual=float(s['residual']);linear.append(residual)
   assert math.isfinite(residual) and residual<=1e-11
 assert len(steps)==m and [int(s['step']) for s in steps]==list(range(1,m+1))
 for key in ('ksp_seconds','linear_solve_seconds'):
  assert math.isclose(sum(float(s[key]) for s in steps),report[key],rel_tol=1e-10)
 assert math.isclose(sum(float(s['step_seconds']) for s in steps),report['iteration_seconds'],rel_tol=1e-10)
 print(f'PASS {label} DOFs={report["dofs"]} T=0.1 steps={m}, timing sums, error orders and memory records')
print(f'PASS {len(labels)}/6 layers; {len(linear)} true linear residuals, maximum={max(linear):.6e}')
