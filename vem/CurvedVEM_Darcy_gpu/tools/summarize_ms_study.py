#!/usr/bin/env python3
"""汇总 MS 主函数输出：误差/阶、实际 DOF、分项计时与逐步内存；支持运行中刷新。"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as file:
        return [r for r in csv.DictReader(file) if None not in r and all(v is not None for v in r.values())]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value, digits=3):
    value = float(value)
    return f'{value:.{digits}f}' if math.isfinite(value) else '—'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    root = args.root
    convergence = read_csv(root/'convergence.csv')
    completed = list(dict.fromkeys(row['mesh'] for row in convergence))
    status = '运行中，以下只列已完成网格；未计算 k=2。'
    summary_file = root/'monitor/summary.json'
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        status = ('全部完成' if summary['returncode']==0 else '运行已停止/失败，未完成网格不报告最终误差')
        status += f"；进程退出码 {summary['returncode']}，监控墙钟 {summary['elapsed_s']:.3f} s。未计算 k=2。"
    steps, memories, timing = [], [], []
    backend_line = ''
    for directory in sorted(root.glob('mesh_*'), key=lambda p:int(p.name.split('_')[1].split('x')[0])):
        label = directory.name[5:]
        logfile = directory/'iterations.log'
        local=[]
        for line in logfile.read_text().splitlines() if logfile.exists() else []:
            if line.startswith('PETSc KSP=') and not backend_line:
                # 从实际求解日志取后端，避免把历史 GPU 因子误标成当前 CPU 因子。
                backend_line = line
            if re.match(r'step=\d+ step_seconds=', line):
                row = dict(field.split('=',1) for field in line.split())
                item = dict(mesh=label, step=int(row['step']), step_seconds=float(row['step_seconds']),
                            ksp_seconds=float(row['ksp_seconds']), linear_solve_seconds=float(row['linear_solve_seconds']),
                            elapsed_seconds=float(row['elapsed_seconds']), rss_mib=int(row['rss_bytes'])/2**20,
                            gpu_used_mib=int(row['gpu_used_bytes'])/2**20,
                            gpu_free_mib=int(row['gpu_free_bytes'])/2**20)
                steps.append(item);local.append(item)
        if local:
            # 首个四分之一时间步视作预热窗口；同时保留完整原始数据供检查。
            warm=local[max(1,len(local)//4):] or local
            memories.append(dict(mesh=label, sampled_steps=len(local), warm_start_step=warm[0]['step'],
                                 rss_first_mib=local[0]['rss_mib'],rss_peak_mib=max(x['rss_mib'] for x in local),
                                 rss_final_mib=local[-1]['rss_mib'],rss_warm_delta_mib=warm[-1]['rss_mib']-warm[0]['rss_mib'],
                                 rss_warm_range_mib=max(x['rss_mib'] for x in warm)-min(x['rss_mib'] for x in warm),
                                 gpu_first_mib=local[0]['gpu_used_mib'],gpu_peak_mib=max(x['gpu_used_mib'] for x in local),
                                 gpu_final_mib=local[-1]['gpu_used_mib'],gpu_warm_delta_mib=warm[-1]['gpu_used_mib']-warm[0]['gpu_used_mib'],
                                 gpu_warm_range_mib=max(x['gpu_used_mib'] for x in warm)-min(x['gpu_used_mib'] for x in warm)))
        if label in completed:
            row=next(x for x in convergence if x['mesh']==label)
            count=int(row['steps'])
            timing.append(dict(mesh=label,cells=int(row['cells']),dofs=int(row['dofs']),steps=count,dt=float(row['dt']),
                               average_step_seconds=float(row['average_step_seconds']),ksp_seconds=float(row['ksp_seconds']),
                               average_ksp_seconds=float(row['ksp_seconds'])/count,
                               linear_solve_seconds=float(row['linear_solve_seconds']),
                               total_seconds=float(row['total_seconds']),linear_iterations=int(row['linear_iterations'])))
    write_csv(root/'step_metrics.csv',steps)
    write_csv(root/'memory_summary.csv',memories)
    write_csv(root/'timing_summary.csv',timing)
    lines=['# MS4 k=1 六层运行汇总','',status,'',
           '更新时间：' + datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'',
           '实际代数后端：`' + (backend_line or '尚未完成首次线性求解') + '`','',
           '网格文件编号为 {2,4,8,16,32,64}，实际为 4×4 至 128×128；BDF2、T=0.1、dt=0.1h。',
           'k=1 四边形每组分：通量自由度 2N_edges+3N_cells，浓度自由度 3N_cells；三组分总数为 6N_edges+18N_cells=30m²+12m（m×m 网格），含边界约束自由度。','',
           '平均每步指时间推进迭代耗时/步数，含系数更新、右端、Picard 与代数流程，不含初始网格准备和误差积分；KSP 包括预条件器准备、迭代与必要校正。总耗时还含准备、诊断、保存和释放。','',
           '| 实际网格 | 单元数 | 总自由度 | 步数 | dt | 平均每步 s | KSP 总计 s | 平均每步 KSP s | 总耗时 s |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in timing:
        lines.append(f"| {r['mesh']} | {r['cells']} | {r['dofs']} | {r['steps']} | {r['dt']:.8g} | {r['average_step_seconds']:.3f} | {r['ksp_seconds']:.3f} | {r['average_ksp_seconds']:.3f} | {r['total_seconds']:.3f} |")
    lines += ['', '## 计算域绝对 L² 误差与收敛阶','',
              '各层在同一个 T=0.1 比较。阶为 log(e旧/e新)/log(h旧/h新)，不截断低阶、负阶；物理域误差和散度误差见 convergence.csv。','',
              '| 网格 | 组分 | 浓度 L² | 阶 | 通量 L² | 阶 |', '|---|---:|---:|---:|---:|---:|']
    for r in convergence:
        lines.append(f"| {r['mesh']} | {int(r['component'])+1} | {float(r['scalar_error']):.6e} | {f(r['scalar_order'])} | {float(r['flux_error']):.6e} | {f(r['flux_order'])} |")
    lines += ['', '## 每步结束时的内存/显存','',
              'RSS 为当前求解进程的常驻内存；显存是 cudaMemGetInfo 的整张设备口径，可能包含桌面/其他程序。预热窗口取前四分之一时间步（至少首步），以下变化为窗口之后末值减首值。应结合时间曲线判断，不能将网格加密或分配器缓存直接认定为泄漏。','',
              '| 网格 | 已采样步数 | RSS 峰值 MiB | RSS 预热后变化 MiB | 显存峰值 MiB | 显存预热后变化 MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for r in memories:
        lines.append(f"| {r['mesh']} | {r['sampled_steps']} | {r['rss_peak_mib']:.2f} | {r['rss_warm_delta_mib']:+.2f} | {r['gpu_peak_mib']:.2f} | {r['gpu_warm_delta_mib']:+.2f} |")
    lines += ['', '完整数据：step_metrics.csv（逐步计时/内存）、memory_summary.csv、timing_summary.csv、monitor/resources.csv（每 5 秒采样）、monitor/program.log（完整 stdout），各网格 iterations.log 保存线性真实残差与逐步记录。',
              'results.txt 的 memory phase=released 行记录该网格方程/求解器析构后的资源；CPU 网格与进程级 CUDA/PETSc 上下文当时仍存在。','']
    if args.plot:
        os.environ.setdefault('MPLCONFIGDIR','/tmp/vem_matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        samples=read_csv(root/'monitor/resources.csv')
        if samples:
            fig,axes=plt.subplots(2,1,sharex=True,figsize=(10,6),layout='constrained')
            labels=list(dict.fromkeys(r['mesh'] for r in samples))
            for ax,key,label in zip(axes,['rss_mib','gpu_used_mib'],['Process RSS (MiB)','Device GPU memory (MiB)']):
                for group in labels:
                    if group=='startup': continue
                    subset=[r for r in samples if r['mesh']==group]
                    values=[float(r[key]) if r[key] else math.nan for r in subset]
                    ax.plot([float(r['elapsed_s'])/60 for r in subset],values,linewidth=1,label=group)
                ax.set_ylabel(label);ax.grid(alpha=.25)
            axes[0].legend(loc='upper left',ncols=3,fontsize=8)
            axes[-1].set_xlabel('Elapsed time (minutes)')
            fig.savefig(root/'memory_history.png',dpi=170)
            fig.savefig(root/'memory_history.svg')
            plt.close(fig)
            lines += ['![运行内存曲线](memory_history.png)','']
    (root/'运行汇总.md').write_text('\n'.join(lines))
    print(f'Updated {root}/运行汇总.md: completed {len(completed)}/6, step records {len(steps)}')


if __name__=='__main__':
    main()
