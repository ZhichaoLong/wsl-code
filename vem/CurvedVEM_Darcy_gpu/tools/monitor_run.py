#!/usr/bin/env python3
"""启动一个算例并持续记录进程 RSS、系统可用内存及设备总显存；不设时间截断。"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


def proc_fields(path):
    """读取 Linux /proc 中以 kB 为单位的内存项；进程退出时允许读不到。"""
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(':')
            parts = value.split()
            if parts and parts[0].isdigit():
                result[key] = int(parts[0])
    except (OSError, ValueError):
        pass
    return result


def gpu_sample():
    """NVML/nvidia-smi 是整张显卡的口径，不能当作本进程独占显存。"""
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits', '-i', '0'],
                           capture_output=True, text=True, timeout=5, check=True)
        return [float(x.strip()) for x in p.stdout.strip().split(',')]
    except (OSError, ValueError, subprocess.SubprocessError):
        return [None, None, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cwd', required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--report-interval', type=float, default=60)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command or args.interval <= 0:
        parser.error('需要命令和正采样间隔')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)  # 防止覆盖此前监控证据。
    start = time.monotonic()
    metadata = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
                    command=command, cwd=args.cwd, interval_s=args.interval,
                    gpu_scope='whole_device', baseline_gpu=gpu_sample())
    process = subprocess.Popen(command, cwd=args.cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    metadata.update(pid=process.pid, monitor_pid=os.getpid())
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    state = dict(mesh='startup', last_line='', reader_error=None)
    print(json.dumps(metadata), flush=True)

    def forward_output():
        try:
            with (output / 'program.log').open('w', buffering=1) as log:
                for line in process.stdout:
                    state['last_line'] = line.strip()
                    match = re.search(r'正在计算 (\S+) 网格=', line)
                    if match:
                        state['mesh'] = match[1]
                    log.write(f'[{time.monotonic()-start:.3f}s] {line}')
        except Exception as error:
            state['reader_error'] = repr(error)

    reader = threading.Thread(target=forward_output, daemon=True)
    reader.start()
    stopped = None

    def stop(signum=None, frame=None):
        nonlocal stopped
        stopped = stopped or f'signal_{signum}'
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    peak_rss = peak_gpu = 0
    last_report = -args.report_interval
    try:
        with (output / 'resources.csv').open('w', buffering=1) as file:
            writer = csv.writer(file)
            writer.writerow(['elapsed_s', 'mesh', 'rss_mib', 'hwm_mib', 'swap_mib',
                             'host_available_mib', 'gpu_used_mib', 'gpu_free_mib', 'gpu_util_pct'])
            while process.poll() is None:
                elapsed = time.monotonic() - start
                status = proc_fields(f'/proc/{process.pid}/status')
                host = proc_fields('/proc/meminfo')
                gpu = gpu_sample()
                rss = status.get('VmRSS', 0) / 1024
                peak_rss = max(peak_rss, rss)
                peak_gpu = max(peak_gpu, gpu[0] or 0)
                writer.writerow([elapsed, state['mesh'], rss, status.get('VmHWM', 0)/1024,
                                 status.get('VmSwap', 0)/1024, host.get('MemAvailable', 0)/1024] + gpu)
                if elapsed-last_report >= args.report_interval:
                    print(f"{elapsed:.1f}s mesh={state['mesh']} RSS={rss:.1f}MiB "
                          f"GPU={gpu[0]}MiB last={state['last_line'][-220:]}", flush=True)
                    last_report = elapsed
                if state['reader_error']:
                    raise RuntimeError(state['reader_error'])
                time.sleep(args.interval)
    finally:
        if process.poll() is None:
            stopped = stopped or 'monitor_exception'
            stop()
        reader.join(timeout=10)
        summary = dict(metadata, returncode=process.returncode, stop_reason=stopped,
                       reader_error=state['reader_error'], elapsed_s=time.monotonic()-start,
                       peak_rss_mib=peak_rss, peak_gpu_used_mib=peak_gpu, final_gpu=gpu_sample())
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    return 0 if process.returncode == 0 and state['reader_error'] is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
