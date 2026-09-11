# 与 CPU 工程相同：main/*_main.cpp 自动生成独立可执行文件，make 只编译。
# 算例、网格序列、时间尺度和输出名称均在各主函数中修改。
CUDA_HOME ?= /usr/local/cuda-13.0
NVCC = $(CUDA_HOME)/bin/nvcc
CXX = mpicxx
PETSC_DIR ?= /home/lzccode/petsc
PETSC_ARCH ?= arch-linux-c-opt
PETSC_FLAGS = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
PETSC_LIBS = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -Xlinker -rpath -Xlinker $(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
ARCH ?= sm_120
CPPFLAGS = $(PETSC_FLAGS) -Iinclude -I. -I$(CUDA_HOME)/include
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -MMD -MP
NVFLAGS = -ccbin $(CXX) -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler=-Wall,-Wextra -MMD -MP

HOST = $(wildcard src/*.cpp mesh/*.cpp core/*.cpp lib/*.cpp examples/*.cpp)
GPU = $(wildcard src/*.cu)
OBJ = $(HOST:%.cpp=build/%.o) $(GPU:%.cu=build/%.o)
MAIN_SOURCES = $(wildcard main/*_main.cpp)
MAIN_NAMES = $(patsubst main/%_main.cpp,%,$(MAIN_SOURCES))
MAIN_OBJECTS = $(MAIN_SOURCES:%.cpp=build/%.o)
MAIN_TARGETS = $(addprefix build/,$(MAIN_NAMES))

all: main
main: $(MAIN_TARGETS)
# make MS_curved_bdf2 只编译该入口，不隐式运行长时间算例。
$(MAIN_NAMES): %: build/%

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
build/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVFLAGS) -c $< -o $@
$(MAIN_TARGETS): build/%: build/main/%_main.o $(OBJ)
	$(NVCC) -ccbin $(CXX) -arch=$(ARCH) $^ $(PETSC_LIBS) -o $@

# 默认只运行一个制造解算例的空间实验；旧六例批量功能须显式选择 run-all。
run: build/MS_curved_bdf2
	./build/MS_curved_bdf2
run-all: build/vem_gpu_all
	./build/vem_gpu_all
benchmark: build/vem_gpu_benchmark
	./build/vem_gpu_benchmark
list:
	@echo "独立主程序：$(MAIN_NAMES)"

-include $(OBJ:.o=.d) $(MAIN_OBJECTS:.o=.d) build/tests/test_stage.d
.PHONY: all main run run-all benchmark list $(MAIN_NAMES)

# 临时阶段测试入口；tests/test_stage.cpp 通过后删除，不加入默认构建。
build/test_stage: build/tests/test_stage.o $(OBJ)
	$(NVCC) -ccbin $(CXX) -arch=$(ARCH) $^ $(PETSC_LIBS) -o $@
