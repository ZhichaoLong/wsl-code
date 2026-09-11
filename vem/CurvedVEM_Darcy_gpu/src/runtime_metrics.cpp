/** Linux 进程常驻内存与 CUDA 可用显存的轻量采样，不分配 GPU 工作区。 */
#include "runtime_metrics.h"
#include "device.cuh"
#include <fstream>
#include <sstream>
namespace vgpu {
MemorySnapshot sample_memory() {
    MemorySnapshot result;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::istringstream value(line.substr(6));
            size_t kib = 0;
            value >> kib;
            result.rss_bytes = kib * 1024;
            break;
        }
    }
    size_t total = 0;
    CUDA(cudaMemGetInfo(&result.gpu_free_bytes, &total));
    result.gpu_used_bytes = total - result.gpu_free_bytes;
    return result;
}
}
