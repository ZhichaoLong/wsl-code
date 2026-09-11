/**
 * PETSc 对象唯一所有权和 CUDA 数组借用。
 * data() 开始设备数组借用；restore()/raw()/zero()/download() 结束借用。
 * 借用期间只允许自定义 CUDA 核访问，不允许 PETSc 代数运算。
 * CUDA 核使用默认流；借用交接时同步，保证与 PETSc 自有流的可见性。
 */
#pragma once
#include <petscksp.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <type_traits>

namespace vgpu {
// PETSc 返回码转换为 C++ 异常；C 回调中必须另外捕获，禁止异常跨 C ABI。
inline void petsc_check(PetscErrorCode error, const char* call) {
    if (error)
        throw std::runtime_error(std::string(call) + " PETSc error=" + std::to_string(error));
}
#define PETSC(call) ::vgpu::petsc_check((call), #call)
static_assert(std::is_same<PetscScalar, double>::value, "需要 PETSc 双精度实数构建");

// main 中最先构造、最后析构；只 Finalize 自己初始化的 PETSc/MPI。
class PetscSession {
    bool owner_ = false;

public:
    explicit PetscSession(bool memory_debug = false) {
        PetscBool initialized;
        PETSC(PetscInitialized(&initialized));
        if (!initialized) {
            // 当前工程为单 GPU / COMM_SELF；系统 OpenMPI 没有 GPU-aware 支持。
            // 分配跟踪必须在初始化时启用，不能在已有 PETSc 对象后切换分配器。
            int argc = memory_debug ? 7 : 3;
            char name[] = "vem_gpu", option[] = "-use_gpu_aware_mpi", value[] = "0";
            char malloc_debug[] = "-malloc_debug", malloc_dump[] = "-malloc_dump";
            char objects[] = "-objects_dump", all[] = "all";
            char* args[] = {name, option, value, malloc_debug, malloc_dump, objects, all, nullptr};
            char** argv = args;
            PETSC(PetscInitialize(&argc, &argv, nullptr, nullptr));
            owner_ = true;
        }
        const auto error = PetscPushErrorHandler(PetscReturnErrorHandler, nullptr);
        if (error) {
            if (owner_)
                PetscFinalize();
            PETSC(error);
        }
    }
    ~PetscSession() {
        PetscPopErrorHandler();
        if (owner_)
            PetscFinalize();
    }
    PetscSession(const PetscSession&) = delete;
    PetscSession& operator=(const PetscSession&) = delete;
};

// 所有 PETSc 句柄均以对应 Destroy 释放；不释放 GetPC/GetMatrix 返回的借用子对象。
template <class T, PetscErrorCode (*Destroy)(T*)> class PetscOwner {
    T value_ = nullptr;

public:
    PetscOwner() = default;
    ~PetscOwner() {
        if (value_)
            Destroy(&value_);
    }
    PetscOwner(const PetscOwner&) = delete;
    PetscOwner& operator=(const PetscOwner&) = delete;
    T get() const {
        return value_;
    }
    // 只用于向 Create 函数传出参数；拒绝覆盖仍有所有权的句柄。
    T* put() {
        if (value_)
            throw std::logic_error("PETSc handle already owned");
        return &value_;
    }
    void reset() {
        if (value_)
            PETSC(Destroy(&value_));
    }
};
using PetscMat = PetscOwner<Mat, MatDestroy>;
using PetscKsp = PetscOwner<KSP, KSPDestroy>;
using PetscScatter = PetscOwner<VecScatter, VecScatterDestroy>;
using PetscIs = PetscOwner<IS, ISDestroy>;

// 数值向量和单元稠密矩阵批量存储均由 PETSc 分配显存。
// Dense=true 的 Mat(n,1) 是打包视图，各单元真实行列形状由 Cell 偏移描述；
// 不把此打包 Mat 当作全局算子，也不为数万个单元分别创建 PETSc 小对象。
template <bool Dense> class PetscDeviceStorage {
    using Handle = typename std::conditional<Dense, Mat, Vec>::type;
    Handle object_ = nullptr;
    mutable double* borrowed_ = nullptr;
    size_t size_ = 0;

public:
    PetscDeviceStorage() = default;
    // 委托构造保证 resize 抛异常时，已完成初始化的对象仍执行析构清理。
    explicit PetscDeviceStorage(size_t n) : PetscDeviceStorage() {
        resize(n);
    }
    ~PetscDeviceStorage() {
        // 析构不抛异常；即使同步/Restore 报错，也继续尝试销毁对象。
        if (borrowed_) {
            cudaDeviceSynchronize();
            if constexpr (Dense)
                MatDenseCUDARestoreArray(object_, &borrowed_);
            else
                VecCUDARestoreArray(object_, &borrowed_);
        }
        if (object_) {
            if constexpr (Dense)
                MatDestroy(&object_);
            else
                VecDestroy(&object_);
        }
    }
    PetscDeviceStorage(const PetscDeviceStorage&) = delete;
    PetscDeviceStorage& operator=(const PetscDeviceStorage&) = delete;

    // 专用 CUDA 核写入结束后交还 PETSc；重复调用没有额外同步。
    void restore() const {
        if (!borrowed_)
            return;
        auto error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        if constexpr (Dense)
            PETSC(MatDenseCUDARestoreArray(object_, &borrowed_));
        else
            PETSC(VecCUDARestoreArray(object_, &borrowed_));
    }
    Handle raw() const {
        restore();
        return object_;
    }
    // 改变长度时释放旧对象，不保留旧值；n=0 用于主动释放诊断矩阵。
    void resize(size_t n) {
        if (n == size_)
            return;
        if (n > size_t(std::numeric_limits<PetscInt>::max()))
            throw std::overflow_error("PETSc local storage overflow");
        restore();
        if (object_) {
            if constexpr (Dense)
                PETSC(MatDestroy(&object_));
            else
                PETSC(VecDestroy(&object_));
        }
        size_ = 0;
        if (n) {
            if constexpr (Dense)
                PETSC(MatCreateSeqDenseCUDA(PETSC_COMM_SELF, PetscInt(n), 1, nullptr, &object_));
            else
                PETSC(VecCreateSeqCUDA(PETSC_COMM_SELF, PetscInt(n), &object_));
        }
        size_ = n;
    }
    // 多个核可复用同一借用；任何 PETSc 运算前都必须 raw()/restore()。
    double* data() const {
        if (!object_)
            return nullptr;
        if (!borrowed_) {
            if constexpr (Dense)
                PETSC(MatDenseCUDAGetArray(object_, &borrowed_));
            else
                PETSC(VecCUDAGetArray(object_, &borrowed_));
            auto error = cudaDeviceSynchronize();
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
        }
        return borrowed_;
    }
    size_t size() const {
        return size_;
    }
    void zero() {
        if (!object_)
            return;
        restore();
        if constexpr (Dense)
            PETSC(MatZeroEntries(object_));
        else
            PETSC(VecSet(object_, 0));
    }
    // CPU 初始数据上传；常规时间步不经过此接口下载/上传解向量。
    void upload(const std::vector<double>& values) {
        resize(values.size());
        if (size_) {
            auto error = cudaMemcpy(data(), values.data(), size_ * sizeof(double), cudaMemcpyHostToDevice);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
            restore();
        }
    }
    // 仅供保存解、少量报告或数值回归使用。
    std::vector<double> download() const {
        std::vector<double> values(size_);
        if (size_) {
            auto error = cudaMemcpy(values.data(), data(), size_ * sizeof(double), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
            restore();
        }
        return values;
    }
};
using PetscVector = PetscDeviceStorage<false>;
using PetscDense = PetscDeviceStorage<true>;
} // namespace vgpu
