# CurvedVEM Darcy - 自动构建脚本
# tests/*_main.cpp 自动识别为测试入口，其余模块 .cpp 自动识别为公共源码。

# PETSc 配置：可由环境变量或命令行覆盖。
PETSC_DIR ?= /home/lzccode/petsc
PETSC_ARCH ?= arch-linux-c-opt

# 编译器与选项。所有目标统一使用 MPI/PETSc，不再区分 PETSc 测试。
CXX = mpicxx
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
DEPFLAGS = -MMD -MP
CPPFLAGS = -I. -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
LDFLAGS = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib \
          -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib
LDLIBS = -lpetsc -lm

BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
TEST_DIR = tests
MAIN_DIR = main
TOOL_DIR = tools
SOURCE_DIRS = core examples mesh lib solver

# 自动发现公共模块与所有以 _main.cpp 结尾的测试入口。
COMMON_SRCS := $(foreach dir,$(SOURCE_DIRS),$(wildcard $(dir)/*.cpp))
TEST_SRCS := $(wildcard $(TEST_DIR)/*_main.cpp)
TEST_NAMES := $(patsubst $(TEST_DIR)/%_main.cpp,%,$(TEST_SRCS))
NORMAL_TEST_NAMES := $(filter-out test_hdiv_matrix,$(TEST_NAMES))

# 主程序配置：main/ 下每个 *_main.cpp 单独生成一个可执行文件
MAIN_SRCS := $(wildcard $(MAIN_DIR)/*_main.cpp)
MAIN_NAMES := $(patsubst $(MAIN_DIR)/%_main.cpp,%,$(MAIN_SRCS))
MAIN_TARGETS := $(addprefix $(BUILD_DIR)/,$(MAIN_NAMES))

# 工具程序配置：tools/ 下每个 *.cpp 单独生成一个可执行文件
TOOL_SRCS := $(wildcard $(TOOL_DIR)/*.cpp)
TOOL_NAMES := $(patsubst $(TOOL_DIR)/%.cpp,%,$(TOOL_SRCS))
TOOL_TARGETS := $(addprefix $(BUILD_DIR)/tools/,$(TOOL_NAMES))

OLD_PROJECT_DIR = ../MS_FVM
OLD_HDIV_DIR = $(OLD_PROJECT_DIR)/vem_Hdiv
OLD_REFERENCE_SRC = $(TEST_DIR)/reference/old_hdiv_reference_main.cpp
OLD_REFERENCE_TARGET = $(BUILD_DIR)/old_hdiv_reference
OLD_REFERENCE_DATA = $(BUILD_DIR)/old_hdiv_reference.txt
OLD_REFERENCE_SRCS = $(OLD_REFERENCE_SRC) \
                     $(OLD_HDIV_DIR)/HdivMatrix.cpp \
                     $(OLD_HDIV_DIR)/BasisFunctionPloy.cpp \
                     $(OLD_PROJECT_DIR)/MeshReader.cpp \
                     $(OLD_PROJECT_DIR)/triangle_mesher.cpp
OLD_CPPFLAGS = $(CPPFLAGS) -I$(OLD_HDIV_DIR) -I$(OLD_PROJECT_DIR)

# 保留源码目录层级，避免不同目录中同名源文件的目标文件冲突。
COMMON_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRCS))
TEST_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))
TEST_TARGETS := $(addprefix $(BUILD_DIR)/,$(TEST_NAMES))
DEPS := $(COMMON_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(MAIN_OBJS:.o=.d)

MAIN_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(MAIN_SRCS))
TOOL_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TOOL_SRCS))
DEPS += $(TOOL_OBJS:.o=.d)

.PHONY: all compile test tools clean list $(TEST_NAMES) $(MAIN_NAMES) $(TOOL_NAMES) \
        test_quad test_darcy test_mesh test_triangulator test_basis test_petsc

# make 默认只编译；make test 编译并运行所有自动发现的测试。
all: compile main tools

compile: $(TEST_TARGETS)
	@echo "所有测试程序编译完成：$(TEST_TARGETS)"

# 主程序目标
main: $(MAIN_TARGETS)
	@echo "所有主程序编译完成：$(MAIN_TARGETS)"

# 每个主程序由自己的 main 目标文件和所有公共模块目标文件链接而成
$(MAIN_TARGETS): $(BUILD_DIR)/%: $(OBJ_DIR)/$(MAIN_DIR)/%_main.o $(COMMON_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "链接生成主程序：$@"

# 工具程序目标
tools: $(TOOL_TARGETS)
	@echo "所有工具程序编译完成：$(TOOL_TARGETS)"

# 每个工具程序由自己的源文件和所有公共模块目标文件链接而成
$(TOOL_TARGETS): $(BUILD_DIR)/tools/%: $(OBJ_DIR)/$(TOOL_DIR)/%.o $(COMMON_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "链接生成工具：$@"

test: $(NORMAL_TEST_NAMES) test_hdiv_matrix
	@echo "全部测试运行完成"

# 每个测试程序由自己的 main 目标文件和所有公共模块目标文件链接而成。
$(TEST_TARGETS): $(BUILD_DIR)/%: $(OBJ_DIR)/$(TEST_DIR)/%_main.o $(COMMON_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "链接生成：$@"

# 通用编译规则，同时生成头文件依赖文件。
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@
	@echo "编译完成：$@"

# 自动生成与测试文件同名的运行目标，例如 make test_polynomial_basis。
$(NORMAL_TEST_NAMES): %: $(BUILD_DIR)/%
	@echo "========================================"
	@echo "  运行 $@ ..."
	@echo "========================================"
	@./$<

# HdivMatrix 使用独立旧实现可执行文件实时生成参考结果，避免同名类冲突。
$(OLD_REFERENCE_TARGET): $(OLD_REFERENCE_SRCS)
	@mkdir -p $(dir $@)
	$(CXX) $(OLD_CPPFLAGS) $(CXXFLAGS) -Wno-unused-variable -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "链接生成旧 HdivMatrix 参考程序：$@"

.PHONY: test_hdiv_matrix
test_hdiv_matrix: $(BUILD_DIR)/test_hdiv_matrix $(OLD_REFERENCE_TARGET)
	@echo "========================================"
	@echo "  运行新旧 HdivMatrix 对照测试 ..."
	@echo "========================================"
	@$(OLD_REFERENCE_TARGET) $(OLD_REFERENCE_DATA) mesh/data
	@./$(BUILD_DIR)/test_hdiv_matrix $(OLD_REFERENCE_DATA) mesh/data

# 兼容此前常用的简短测试目标；新增测试无需修改 Makefile。
test_quad: test_quadrature
test_darcy: test_darcy_problem
test_mesh: test_straight_mesh
test_triangulator: test_polygon_triangulator
test_basis: test_polynomial_basis
test_petsc: test_petsc_utils

list:
	@echo "自动发现的公共源码：$(COMMON_SRCS)"
	@echo "自动发现的测试入口：$(TEST_SRCS)"
	@echo "将生成的测试程序：$(TEST_TARGETS)"
	@echo "自动发现的主程序入口：$(MAIN_SRCS)"
	@echo "将生成的主程序：$(MAIN_TARGETS)"
	@echo "自动发现的工具程序：$(TOOL_SRCS)"
	@echo "将生成的工具：$(TOOL_TARGETS)"
	@echo "公共模块目标文件：$(COMMON_OBJS)"

clean:
	rm -rf $(BUILD_DIR)
	@echo "已清理 build 目录"

-include $(DEPS)
