cmake_minimum_required(VERSION 3.20)

project(cuda_linear_solver LANGUAGES CXX CUDA)

# 设置二进制文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/main_aux)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RTX 5060 的计算能力是 8.9 (Ada Lovelace 架构)
set(CMAKE_CUDA_ARCHITECTURES 89)

# 可选：调试/Release 编译设置
# set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
# set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit Include Dir: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA Toolkit Library Dir: ${CUDAToolkit_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "Cannot find CUDA Toolkit")
endif()

# ============================================================================
# PETSc 配置
# ============================================================================
set(PETSC_DIR "/home/lzccode/petsc/arch-linux-c-opt" CACHE PATH "PETSc installation directory")

set(PETSC_FOUND FALSE)
set(PETSC_LIBRARIES "")
if(EXISTS "${PETSC_DIR}")
    message(STATUS "Found PETSc at: ${PETSC_DIR}")
    set(PETSC_INCLUDE_DIRS "${PETSC_DIR}/include;/home/lzccode/petsc/include;/usr/lib/x86_64-linux-gnu/openmpi/include")
    set(PETSC_LIB_DIR "${PETSC_DIR}/lib")

    find_library(PETSC_LIBRARY NAMES petsc PATHS "${PETSC_LIB_DIR}" NO_DEFAULT_PATH)

    if(PETSC_LIBRARY)
        message(STATUS "Found PETSc library: ${PETSC_LIBRARY}")
        list(APPEND PETSC_LIBRARIES "${PETSC_LIBRARY}")
        set(PETSC_FOUND TRUE)
    else()
        message(WARNING "PETSc library not found in ${PETSC_LIB_DIR}")
    endif()

    # 查找 MPI 库
    find_library(MPI_LIBRARY NAMES mpi PATHS /usr/lib/x86_64-linux-gnu/openmpi/lib /usr/lib /usr/local/lib)
    if(MPI_LIBRARY)
        list(APPEND PETSC_LIBRARIES "${MPI_LIBRARY}")
    endif()

    # 查找 LAPACK/BLAS，找不到也没关系
    find_library(LAPACK_LIBRARY NAMES lapack PATHS /usr/lib /usr/local/lib)
    if(LAPACK_LIBRARY)
        list(APPEND PETSC_LIBRARIES "${LAPACK_LIBRARY}")
    endif()
    find_library(BLAS_LIBRARY NAMES openblas blas PATHS /usr/lib /usr/local/lib)
    if(BLAS_LIBRARY)
        list(APPEND PETSC_LIBRARIES "${BLAS_LIBRARY}")
    endif()
else()
    message(WARNING "PETSc directory not found at ${PETSC_DIR}, PETSc targets will not be built")
endif()

# ============================================================================
# 公共模块（不含 main 函数的文件）
# ============================================================================
# 这里列出所有公共的 .cu 和 .cpp 文件，它们会被链接到每个可执行文件中
set(COMMON_SRCS
    setgpu.cu
)

# ============================================================================
# 自动扫描并添加可执行文件
#
# 规则：
#   - 所有 .cu 文件：如果文件名以 _main.cu 或 _example.cu 结尾，视为可执行文件
#   - 所有 .cpp 文件：如果文件名以 _main.cpp 或 _example.cpp 结尾，视为可执行文件
#
# ============================================================================

# 扫描所有 .cu 和 .cpp 文件
file(GLOB ALL_CU_SRCS "*.cu")
file(GLOB ALL_CPP_SRCS "*.cpp")

# 从公共模块中排除（避免重复编译）
foreach(common_src ${COMMON_SRCS})
    list(REMOVE_ITEM ALL_CU_SRCS "${CMAKE_SOURCE_DIR}/${common_src}")
    list(REMOVE_ITEM ALL_CU_SRCS "${common_src}")
endforeach()

# 处理 .cu 可执行文件
message(STATUS "========================================================================")
message(STATUS "扫描 CUDA 可执行文件...")
foreach(cu_src ${ALL_CU_SRCS})
    get_filename_component(filename ${cu_src} NAME)
    get_filename_component(target_name ${cu_src} NAME_WE)

    # 检查文件名是否以 _main.cu 或 _example.cu 结尾
    if(filename MATCHES "_main\\.cu$" OR filename MATCHES "_example\\.cu$")
        message(STATUS "  - 发现 CUDA 可执行文件: ${filename} -> ${target_name}")

        add_executable(${target_name} ${cu_src} ${COMMON_SRCS})
        target_include_directories(${target_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${target_name} PRIVATE CUDA::cudart)

        # 如果 PETSc 可用，也链接 PETSc
        if(PETSC_FOUND)
            target_include_directories(${target_name} PRIVATE ${PETSC_INCLUDE_DIRS})
            target_link_libraries(${target_name} PRIVATE ${PETSC_LIBRARIES} m)
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${PETSC_LIB_DIR}"
                INSTALL_RPATH "${PETSC_LIB_DIR}"
            )
        endif()
    endif()
endforeach()

# 处理 .cpp 可执行文件（可能使用 PETSc）
message(STATUS "------------------------------------------------------------------------")
message(STATUS "扫描 C++ 可执行文件...")
foreach(cpp_src ${ALL_CPP_SRCS})
    get_filename_component(filename ${cpp_src} NAME)
    get_filename_component(target_name ${cpp_src} NAME_WE)

    # 检查文件名是否以 _main.cpp 或 _example.cpp 结尾
    if(filename MATCHES "_main\\.cpp$" OR filename MATCHES "_example\\.cpp$")
        message(STATUS "  - 发现 C++ 可执行文件: ${filename} -> ${target_name}")

        add_executable(${target_name} ${cpp_src} ${COMMON_SRCS})
        target_include_directories(${target_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${target_name} PRIVATE CUDA::cudart)

        # 如果 PETSc 可用，也链接 PETSc
        if(PETSC_FOUND)
            target_include_directories(${target_name} PRIVATE ${PETSC_INCLUDE_DIRS})
            target_link_libraries(${target_name} PRIVATE ${PETSC_LIBRARIES} m)
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${PETSC_LIB_DIR}"
                INSTALL_RPATH "${PETSC_LIB_DIR}"
            )
        endif()

        # 允许 .cpp 文件调用 CUDA 运行时
        set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endforeach()

message(STATUS "========================================================================")

# ============================================================================
# Google Test 单元测试（可选）
# ============================================================================
find_package(GTest QUIET)
if (GTest_FOUND)
    message(STATUS "Google Test found, building unit tests if available")
    enable_testing()

    # 查找测试文件
    file(GLOB TEST_SRCS "*_test.cpp" "*_test.cu")
    foreach(test_src ${TEST_SRCS})
        get_filename_component(target_name ${test_src} NAME_WE)
        message(STATUS "  - 发现测试文件: ${test_src} -> ${target_name}")

        add_executable(${target_name} ${test_src} ${COMMON_SRCS})
        target_include_directories(${target_name} PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GTEST_INCLUDE_DIRS}
        )
        target_link_libraries(${target_name} PRIVATE
            CUDA::cudart
            GTest::GTest
            GTest::Main
        )
        set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        add_test(NAME ${target_name} COMMAND ${target_name})
   endforeach()
else()
    message(STATUS "Google Test not found, skipping unit tests")
endif()

# ============================================================================
# 辅助：显示构建信息
# ============================================================================
message(STATUS "")
message(STATUS "构建配置摘要:")
message(STATUS "  输出目录: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "  PETSc 支持: ${PETSC_FOUND}")
message(STATUS "  公共模块: ${COMMON_SRCS}")
message(STATUS "")
