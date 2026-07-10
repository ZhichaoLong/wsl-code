#ifndef TRIANGLE_MESHER_H
#define TRIANGLE_MESHER_H

// 引入原有网格数据结构
#include "MeshReader.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// 三角形结构体,这里的构造函数自动生成了面积，但是只是在创建的时候有正确的面积，如果修改顶点，那么就没法自动更新面积了。
struct Triangle {
    // 三角形三个顶点的全局编号
    std::vector<int> nodes;
    // 三个顶点坐标 [x0,y0, x1,y1, x2,y2]
    std::vector<double> vertices;
    // 三角形面积
    double area = 0.0;

    // 构造函数（C++11及以上）
    Triangle() : nodes(3, 0), vertices(6, 0.0) {}
    
    Triangle(int n0, int n1, int n2, 
             double x0, double y0, double x1, double y1, double x2, double y2) 
        : nodes({n0, n1, n2}), vertices({x0, y0, x1, y1, x2, y2}) {
        area = calculateArea();
    }

    // 计算当前三角形面积（成员函数）
    double calculateArea() const {
        return 0.5 * std::fabs((vertices[2] - vertices[0]) * (vertices[5] - vertices[1]) - 
                              (vertices[4] - vertices[0]) * (vertices[3] - vertices[1]));
    }
};
struct TriangulatedElement {
    // 剖分得到的三角形数量（可通过triangles.size()替代）
    // 存储所有三角形（完全替代C的动态数组）
    std::vector<Triangle> triangles;
    // 清空数据（C++风格）
    void clear() {
        triangles.clear();
    }
};

class TriangleMesher {
public:
    TriangleMesher() = default;
    ~TriangleMesher() = default;
    TriangleMesher(const TriangleMesher&) = delete;
    TriangleMesher& operator=(const TriangleMesher&) = delete;
    TriangleMesher(TriangleMesher&&) = default;
    TriangleMesher& operator=(TriangleMesher&&) = default;
    // 对指定单元进行三角剖分
    TriangulatedElement triangulateElement(const MeshData& meshData, int element_id);

private:
    // 验证单元数据合法性
    bool validateElement(const MeshData& meshData, int element_id) const;
    //判断点是否在三角形内（一维数组传递坐标）
    //  points 单元顶点的一维坐标数组 [x0,y0,x1,y1,...]
    //  p_idx 待判断点的索引
    //  a_idx/b_idx/c_idx 三角形的三个顶点索引
    bool isPointInsideTriangle(const std::vector<double>& points,int p_idx, int a_idx, int b_idx, int c_idx) const;
    //判断是不是耳朵
    bool isEar(int i, const std::vector<double>& points, int num_points) const;
    //耳切法三角剖分
    TriangulatedElement earClipping(const MeshData& meshData, int element_id);
};



#endif // TRIANGLE_MESHER_H