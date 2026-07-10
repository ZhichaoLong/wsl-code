#ifndef ERROR_TPFA_1D_H
#define ERROR_TPFA_1D_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "GaussQuadrature.h"

/**
 * @brief 存储有限体积法解的数据类
 *
 * 存储从txt文件读取的解数据，包括：
 * - 网格单元数
 * - 组分数
 * - 每个组分的单元平均值
 * - 每个组分的通量（在边界点上）
 * - 网格边界和单元信息
 */
class SolutionData {
public:
    int n_cells;                    // 网格单元数
    int n_components;               // 组分数
    double xmin;                    // 左边界
    double xmax;                    // 右边界
    double dx;                      // 单元长度（均匀网格）
    std::vector<double> cell_centers;  // 单元中心坐标
    std::vector<double> face_centers;  // 面中心坐标（n_cells + 1 个）
    std::vector<double> u_avg;     // 单元平均值 [comp * n_cells + cell]
    std::vector<double> flux;      // 通量 [comp * (n_cells + 1) + face]

    SolutionData() : n_cells(0), n_components(0), xmin(0.0), xmax(1.0), dx(0.0) {}

    /**
     * @brief 从txt文件读取解数据（包含平均值和通量）
     * @param filename 文件名
     * @param xmin_in 左边界（如果文件中没有提供）
     * @param xmax_in 右边界（如果文件中没有提供）
     */
    void readFromFile(const std::string& filename, double xmin_in = 0.0, double xmax_in = 1.0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        bool data_started = false;

        // 先读取头信息
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                // 解析头信息
                if (line.find("网格单元数") != std::string::npos) {
                    size_t pos = line.find(":");
                    if (pos != std::string::npos) {
                        n_cells = std::stoi(line.substr(pos + 1));
                    }
                } else if (line.find("组分数") != std::string::npos) {
                    size_t pos = line.find(":");
                    if (pos != std::string::npos) {
                        n_components = std::stoi(line.substr(pos + 1));
                    }
                }
                continue;
            }
            // 遇到非注释行且已解析完头信息，开始读数据
            data_started = true;
            break;
        }

        if (n_cells <= 0 || n_components <= 0) {
            throw std::runtime_error("Invalid header information in file: " + filename);
        }

        // 设置网格信息
        xmin = xmin_in;
        xmax = xmax_in;
        dx = (xmax - xmin) / n_cells;

        // 生成单元中心坐标
        cell_centers.resize(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            cell_centers[i] = xmin + (i + 0.5) * dx;
        }

        // 生成面中心坐标（n_cells + 1 个面）
        face_centers.resize(n_cells + 1);
        for (int i = 0; i <= n_cells; ++i) {
            face_centers[i] = xmin + i * dx;
        }

        // 分配空间
        u_avg.resize(n_components * n_cells);
        flux.resize(n_components * (n_cells + 1));

        // 重新定位到数据开始的地方（如果已经读了一行数据）
        if (data_started && !line.empty()) {
            // 把刚才读的一行放回去
            file.seekg(-(static_cast<std::streamoff>(line.length()) + 1), std::ios_base::cur);
        }

        // 读取数据行
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;

            // 按逗号分隔
            while (std::getline(iss, token, ',')) {
                // 去掉首尾空格
                size_t start = token.find_first_not_of(" \t");
                size_t end = token.find_last_not_of(" \t");
                if (start != std::string::npos && end != std::string::npos) {
                    tokens.push_back(token.substr(start, end - start + 1));
                }
            }

            if (tokens.size() >= 5) {
                int type = std::stoi(tokens[2]);
                int comp = std::stoi(tokens[3]);
                int local_idx = std::stoi(tokens[4]);
                double value = std::stod(tokens[1]);

                if (type == 0) {  // 平均值
                    if (comp >= 0 && comp < n_components && local_idx >= 0 && local_idx < n_cells) {
                        u_avg[comp * n_cells + local_idx] = value;
                    }
                } else if (type == 1) {  // 通量
                    // 通量的本地索引直接就是面索引！
                    // 对于 n_cells 个单元，有 n_cells + 1 个面
                    // 本地索引 0 ~ n_cells 对应物理面 0 ~ n_cells
                    if (comp >= 0 && comp < n_components && local_idx >= 0 && local_idx <= n_cells) {
                        flux[comp * (n_cells + 1) + local_idx] = value;
                    }
                }
            }
        }

        file.close();
        std::cout << "Read solution from " << filename << ": n_cells=" << n_cells
                  << ", n_components=" << n_components << std::endl;
    }

    /**
     * @brief 从新格式的txt文件读取解数据（仅单元平均值，不包含通量）
     * @param filename 文件名
     * @param xmin_in 左边界（如果文件中没有提供）
     * @param xmax_in 右边界（如果文件中没有提供）
     */
    void readFromFileNew(const std::string& filename, double xmin_in = 0.0, double xmax_in = 1.0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        bool data_started = false;

        // 先读取头信息
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                // 解析头信息
                if (line.find("网格单元数") != std::string::npos) {
                    size_t pos = line.find(":");
                    if (pos != std::string::npos) {
                        n_cells = std::stoi(line.substr(pos + 1));
                    }
                } else if (line.find("组分数") != std::string::npos) {
                    size_t pos = line.find(":");
                    if (pos != std::string::npos) {
                        n_components = std::stoi(line.substr(pos + 1));
                    }
                }
                continue;
            }
            // 遇到非注释行且已解析完头信息，开始读数据
            data_started = true;
            break;
        }

        if (n_cells <= 0 || n_components <= 0) {
            throw std::runtime_error("Invalid header information in file: " + filename);
        }

        // 设置网格信息
        xmin = xmin_in;
        xmax = xmax_in;
        dx = (xmax - xmin) / n_cells;

        // 生成单元中心坐标
        cell_centers.resize(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            cell_centers[i] = xmin + (i + 0.5) * dx;
        }

        // 生成面中心坐标
        face_centers.resize(n_cells + 1);
        for (int i = 0; i <= n_cells; ++i) {
            face_centers[i] = xmin + i * dx;
        }

        // 分配空间
        u_avg.resize(n_components * n_cells);
        flux.resize(n_components * (n_cells + 1), 0.0);

        // 重新定位到数据开始的地方（如果已经读了一行数据）
        if (data_started && !line.empty()) {
            // 把刚才读的一行放回去
            file.seekg(-(static_cast<std::streamoff>(line.length()) + 1), std::ios_base::cur);
        }

        // 读取数据行（新格式：索引, 值, 组分, 单元）
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;

            // 按逗号分隔
            while (std::getline(iss, token, ',')) {
                // 去掉首尾空格
                size_t start = token.find_first_not_of(" \t");
                size_t end = token.find_last_not_of(" \t");
                if (start != std::string::npos && end != std::string::npos) {
                    tokens.push_back(token.substr(start, end - start + 1));
                }
            }

            if (tokens.size() >= 4) {
                // 新格式：索引, 值, 组分, 单元
                int comp = std::stoi(tokens[2]);
                int local_idx = std::stoi(tokens[3]);
                double value = std::stod(tokens[1]);

                if (comp >= 0 && comp < n_components && local_idx >= 0 && local_idx < n_cells) {
                    u_avg[comp * n_cells + local_idx] = value;
                }
            }
        }

        file.close();
        std::cout << "Read solution from " << filename << " (new format): n_cells=" << n_cells
                  << ", n_components=" << n_components << std::endl;
    }

    /**
     * @brief 获取指定组分在指定单元的平均值
     */
    double getUValue(int comp, int cell) const {
        return u_avg[comp * n_cells + cell];
    }

    /**
     * @brief 获取指定组分在指定面的通量
     */
    double getFluxValue(int comp, int face) const {
        return flux[comp * (n_cells + 1) + face];
    }
};

/**
 * @brief TPFA 1D 误差计算类
 *
 * 计算单元平均值和通量的L2误差：
 * - 单元平均值误差：使用与MATLAB相同的逻辑，取相邻细网格的算术平均作为参考解
 * - 通量误差：通量在边界点上，直接在相同的物理面位置比较
 */
class ErrorCalculatorTPFA1D {
public:
    SolutionData coarse_sol;    // 粗网格解
    SolutionData fine_sol;      // 细网格参考解

    ErrorCalculatorTPFA1D() {}

    /**
     * @brief 加载粗网格解
     */
    void loadCoarseSolution(const std::string& filename, double xmin = 0.0, double xmax = 1.0) {
        coarse_sol.readFromFile(filename, xmin, xmax);
    }

    /**
     * @brief 加载细网格参考解
     */
    void loadFineSolution(const std::string& filename, double xmin = 0.0, double xmax = 1.0) {
        fine_sol.readFromFile(filename, xmin, xmax);
    }

    /**
     * @brief 加载粗网格解（新格式）
     */
    void loadCoarseSolutionNew(const std::string& filename, double xmin = 0.0, double xmax = 1.0) {
        coarse_sol.readFromFileNew(filename, xmin, xmax);
    }

    /**
     * @brief 加载细网格参考解（新格式）
     */
    void loadFineSolutionNew(const std::string& filename, double xmin = 0.0, double xmax = 1.0) {
        fine_sol.readFromFileNew(filename, xmin, xmax);
    }

    /**
     * @brief 调和平均（保留用于向后兼容）
     */
    double harmonicMean(double a, double b) const {
        const double eps = 1e-18;
        // 如果任一值小于等于0，返回0（保正）
        if (a <= eps || b <= eps) {
            return 0.0;
        }
        // 如果两值几乎相等，直接返回该值
        if (std::fabs(a - b) < eps) {
            return a;
        }
        // 否则返回调和平均
        return 2.0 * a * b / (a + b);
    }

    /**
     * @brief 在细网格上获取给定点的单元平均值近似值（使用MATLAB相同的逻辑：相邻细网格单元的算术平均）
     * @param x 点的坐标
     * @param comp 组分索引
     * @return 细网格在该点的近似值
     */
    double getFineUValueAt(double x, int comp) const {
        const double eps = 1e-18;

        // 找到第一个 xc_fine >= x 的索引
        int idx = -1;
        for (int i = 0; i < fine_sol.n_cells; ++i) {
            if (fine_sol.cell_centers[i] >= x - eps) {
                idx = i;
                break;
            }
        }

        int idx_left, idx_right;
        if (idx == -1) {
            // 所有细网格中心都小于 x，取最后一个
            idx_left = fine_sol.n_cells - 1;
            idx_right = fine_sol.n_cells - 1;
        } else if (idx == 0) {
            // 第一个细网格中心就大于等于 x
            idx_left = 0;
            idx_right = 0;
        } else {
            idx_left = idx - 1;
            idx_right = idx;
        }

        // 用相邻细网格的平均值作为参考解
        double u_left = fine_sol.getUValue(comp, idx_left);
        double u_right = fine_sol.getUValue(comp, idx_right);
        return (u_left + u_right) / 2.0;
    }

    /**
     * @brief 在细网格上获取指定面位置的通量值
     * @param x 面的坐标
     * @param comp 组分索引
     * @return 细网格在该面的通量值
     */
    double getFineFluxAt(double x, int comp) const {
        const double eps = 1e-18;

        // 通量在面中心上，直接找到坐标匹配的面
        for (int i = 0; i <= fine_sol.n_cells; ++i) {
            if (std::fabs(fine_sol.face_centers[i] - x) < eps) {
                return fine_sol.getFluxValue(comp, i);
            }
        }

        // 如果没有找到精确匹配（粗网格面不在细网格面上），需要找相邻的细网格面
        // 找到第一个细网格面 >= x
        int idx = -1;
        for (int i = 0; i <= fine_sol.n_cells; ++i) {
            if (fine_sol.face_centers[i] >= x - eps) {
                idx = i;
                break;
            }
        }

        int idx_left, idx_right;
        if (idx == -1) {
            idx_left = fine_sol.n_cells;
            idx_right = fine_sol.n_cells;
        } else if (idx == 0) {
            idx_left = 0;
            idx_right = 0;
        } else {
            idx_left = idx - 1;
            idx_right = idx;
        }

        // 通量在面之间用算术平均
        double f_left = fine_sol.getFluxValue(comp, idx_left);
        double f_right = fine_sol.getFluxValue(comp, idx_right);
        return (f_left + f_right) / 2.0;
    }

    /**
     * @brief 计算所有组分的总 L2 误差（仅单元平均值，保持向后兼容）
     * @return 总 L2 误差
     */
    double computeL2Error() const {
        std::vector<double> errors = computeL2ErrorPerComponent();
        double error_sq_sum = 0.0;
        for (double e : errors) {
            error_sq_sum += e * e;
        }
        return std::sqrt(error_sq_sum);
    }

    /**
     * @brief 分别计算每个组分的单元平均值 L2 误差（使用与MATLAB代码相同的逻辑）
     * @return 每个组分的 L2 误差向量
     */
    std::vector<double> computeL2ErrorPerComponent() const {
        std::vector<double> error_sq_sum(coarse_sol.n_components, 0.0);

        // 对粗网格的每个单元进行计算（与MATLAB代码逻辑一致）
        for (int coarse_cell = 0; coarse_cell < coarse_sol.n_cells; ++coarse_cell) {
            // 粗网格单元中心坐标
            double x = coarse_sol.cell_centers[coarse_cell];

            // 对每个组分计算误差
            for (int comp = 0; comp < coarse_sol.n_components; ++comp) {
                // 粗网格在该单元的值
                double u_coarse = coarse_sol.getUValue(comp, coarse_cell);

                // 细网格在该点的参考值（使用相邻细网格的算术平均）
                double u_ref = getFineUValueAt(x, comp);

                // 计算绝对误差平方并累加（与MATLAB一致）
                double diff = u_coarse - u_ref;
                error_sq_sum[comp] += diff * diff * coarse_sol.dx;
            }
        }

        // 开平方得到 L2 误差
        std::vector<double> errors(coarse_sol.n_components);
        for (int comp = 0; comp < coarse_sol.n_components; ++comp) {
            errors[comp] = std::sqrt(error_sq_sum[comp]);
        }

        return errors;
    }

    /**
     * @brief 分别计算每个组分的通量 L2 误差
     * @return 每个组分的通量 L2 误差向量
     */
    std::vector<double> computeL2ErrorFluxPerComponent() const {
        std::vector<double> error_sq_sum(coarse_sol.n_components, 0.0);

        // 对粗网格的每个面进行计算
        for (int coarse_face = 0; coarse_face <= coarse_sol.n_cells; ++coarse_face) {
            // 粗网格面中心坐标
            double x = coarse_sol.face_centers[coarse_face];

            // 对每个组分计算误差
            for (int comp = 0; comp < coarse_sol.n_components; ++comp) {
                // 粗网格在该面的通量值
                double f_coarse = coarse_sol.getFluxValue(comp, coarse_face);

                // 细网格在该面的通量参考值
                double f_ref = getFineFluxAt(x, comp);

                // 计算绝对误差平方并累加
                double diff = f_coarse - f_ref;
                // 通量误差积分：用 dx 作为权重（或者可以用常数权重1，因为点是离散的）
                // 这里使用与单元平均值类似的处理方式
                error_sq_sum[comp] += diff * diff * coarse_sol.dx;
            }
        }

        // 开平方得到 L2 误差
        std::vector<double> errors(coarse_sol.n_components);
        for (int comp = 0; comp < coarse_sol.n_components; ++comp) {
            errors[comp] = std::sqrt(error_sq_sum[comp]);
        }

        return errors;
    }

    /**
     * @brief 打印误差计算结果（分别打印每个组分的单元平均值误差和通量误差）
     */
    void printErrors() const {
        std::vector<double> u_errors = computeL2ErrorPerComponent();
        std::vector<double> flux_errors = computeL2ErrorFluxPerComponent();

        std::cout << "\n========== L2 Error Calculation Results ==========" << std::endl;
        std::cout << "Coarse grid: " << coarse_sol.n_cells << " cells" << std::endl;
        std::cout << "Fine grid (reference): " << fine_sol.n_cells << " cells" << std::endl;
        for (int comp = 0; comp < u_errors.size(); ++comp) {
            std::cout << "  Component " << comp << " - u L2 error: " << u_errors[comp]
                      << ", flux L2 error: " << flux_errors[comp] << std::endl;
        }
        std::cout << "==================================================\n" << std::endl;
    }
};

#endif // ERROR_TPFA_1D_H
