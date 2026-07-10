#include <stdio.h>
//编译指定虚拟架构能力-arch=compute_61 .这里的61就是指计算能力大于6.1的gpu可执行该文件
// 真实架构必须大版本相同的gpu而且真实架构必须大于虚拟计算能力 -code=sum_xy这是指x.y的版本号
__global__ void hello_from_gpu() {//核函数前缀，返回值为void
    // GPU 里必须用 printf，不能用 cout
    printf("✅ Hello World from GPU! block=%d, thread=%d\n", blockIdx.x, threadIdx.x);
}

int main(void) {
    // 获取GPU设备的数量
    // int iDeviceCount = 0;
    // cudaError_t error = cudaGetDeviceCount(&iDeviceCount);
    // if (error != cudaSuccess) {
    //     printf("❌ GPU 错误: %s\n", cudaGetErrorString(error));
    // }else{
    //     printf("✅ GPU 设备数量: %d\n", iDeviceCount);
    // }
    //设置GPU执行使用的设备
    // int iDev = 0;
    // error = cudaSetDevice(iDev);

    //cuda内存管理函数：cudaMalloc cudaMemcpy cudaMemset cudaFree



    // 启动 4 个 block，每个 4 个 thread → 共16个线程
    hello_from_gpu<<<4, 5>>>();//默认一维（网格，线程块）最大网格2^31-1,1024
    //这里有几个默认的索引变量
    // 这里gridDim.x=4.blockDim.x=5.这里是总的值
    // blockIdx.x是索引从0到3
    // threadIdx.x是索引从0到4
    // 每个线程都是唯一id=blockIdx.x*blockDim.x+threadIdx.x，就是一维顺序排列，更高维也是行优先顺序排列
    // 强制等待 GPU 完成 + 刷新输出
    cudaDeviceSynchronize();
    
    // 检查 GPU 是否报错（超级重要！）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ GPU 错误: %s\n", cudaGetErrorString(err));
    }

    printf("✅ Hello from CPU!\n");
    return 0;
}