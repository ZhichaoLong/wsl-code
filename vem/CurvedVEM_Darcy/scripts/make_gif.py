"""
将帧图片合成为 GIF 动图（使用 ffmpeg，内存友好）
用法:
    python make_gif.py <帧目录> [输出gif路径] [fps] [目标宽度]
示例:
    python make_gif.py data/ms_conv_8x8_frames
    python make_gif.py data/ms_conv_8x8_frames data/gif/ms_8x8.gif 10 1200
"""
import sys
import os
import re
import glob
import tempfile
import shutil
import subprocess


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    frame_dir = sys.argv[1].rstrip('/')
    frame_name = os.path.basename(frame_dir)

    out_path = sys.argv[2] if len(sys.argv) > 2 else f'data/gif/{frame_name}.gif'
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    target_width = int(sys.argv[4]) if len(sys.argv) > 4 else 1200

    if not os.path.isdir(frame_dir):
        print(f'错误: 帧目录不存在: {frame_dir}')
        sys.exit(1)

    # 收集所有 png 文件并按帧号排序
    files = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png')))
    if not files:
        print(f'错误: 在 {frame_dir} 中没有找到 frame_*.png')
        sys.exit(1)

    n_frames = len(files)
    print(f'找到 {n_frames} 帧')
    print(f'输出: {out_path}')
    print(f'FPS: {fps}')
    print(f'目标宽度: {target_width}px')

    # 检查 ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except Exception:
        print('错误: 未找到 ffmpeg，请先安装 ffmpeg')
        sys.exit(1)

    # 创建临时目录，按顺序生成软链接（ffmpeg 需要连续编号）
    temp_dir = tempfile.mkdtemp(prefix='gif_temp_')
    try:
        # 按帧号重命名为连续编号
        for i, f in enumerate(files):
            link_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
            # 用硬链接或复制
            shutil.copy(f, link_path)
            if (i + 1) % 100 == 0:
                print(f'  准备中... [{i+1}/{n_frames}]')

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

        # ffmpeg 生成 GIF
        # 先用 palettegen 生成调色板，再用 paletteuse 生成高质量 GIF
        palette_path = os.path.join(temp_dir, 'palette.png')
        input_pattern = os.path.join(temp_dir, 'frame_%05d.png')

        print('  生成调色板...')
        cmd_palette = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-vf', f'fps={fps},scale={target_width}:-1:flags=lanczos,palettegen=max_colors=256',
            palette_path
        ]
        subprocess.run(cmd_palette, capture_output=True, check=True)

        print('  生成 GIF...')
        cmd_gif = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-i', palette_path,
            '-lavfi', f'fps={fps},scale={target_width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3',
            '-loop', '0',
            out_path
        ]
        subprocess.run(cmd_gif, capture_output=True, check=True)

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f'\n完成！共 {n_frames} 帧，大小 {size_mb:.1f} MB')
        print(f'输出文件: {out_path}')

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
