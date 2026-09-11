/** 运行内存采样：进程 RSS 与设备总显存占用，供同网格时间步之间比较。 */
#pragma once
#include <cstddef>
namespace vgpu {
struct MemorySnapshot {
    size_t rss_bytes = 0;
    size_t gpu_used_bytes = 0, gpu_free_bytes = 0;
};
// GPU 数值任务由调用者同步；cudaMemGetInfo 是设备级口径，不是进程专属显存。
MemorySnapshot sample_memory();
}
