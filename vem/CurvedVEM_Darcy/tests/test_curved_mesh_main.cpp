#include <iostream>
#include <cassert>
#include <cmath>
#include "../mesh/straight_mesh.h"
#include "../mesh/curved_mesh.h"
#include "../examples/darcy_problem.h"

int main() {
    std::cout << "Testing CurvedMeshGenerator..." << std::endl;

    // 测试1: 读取直边网格
    std::cout << "\n=== Test 1: Read straight mesh ===" << std::endl;
    vem::StraightMeshReader straight_reader;
    bool success = straight_reader.read_mesh("/home/lzccode/code/vem/gmesh_py/mesh_data/square_4x4.msh");
    assert(success && "Failed to read straight mesh");

    const vem::StraightMeshData& straight_data = straight_reader.get_mesh_data();
    std::cout << "  Straight mesh: " << straight_data.num_nodes << " nodes, "
              << straight_data.num_cells << " cells" << std::endl;

    // 打印直边网格的节点坐标
    std::cout << "  Straight mesh nodes (first 10):" << std::endl;
    for (int i = 0; i < std::min(10, straight_data.num_nodes); ++i) {
        double x = straight_data.node_coords[2 * i];
        double y = straight_data.node_coords[2 * i + 1];
        std::cout << "    Node " << i << ": (" << x << ", " << y << ")" << std::endl;
    }

    // 测试2: 转换为曲边网格数据结构（恒等映射）
    std::cout << "\n=== Test 2: Convert to curved mesh data (identity mapping) ===" << std::endl;
    vem::CurvedMeshData curved_base;
    straight_reader.to_curved_mesh_data(curved_base);

    Darcy::IdentityMapping identity_map;
    vem::CurvedMeshGenerator generator;
    success = generator.generate_from_straight(curved_base, identity_map);
    assert(success && "Failed to generate curved mesh with identity mapping");

    const vem::CurvedMeshData& curved_data_identity = generator.get_mesh_data();
    std::cout << "  Curved mesh (identity): " << curved_data_identity.num_nodes << " nodes" << std::endl;

    // 验证恒等映射后坐标应该不变
    std::cout << "  Verifying identity mapping..." << std::endl;
    for (int i = 0; i < curved_data_identity.num_nodes; ++i) {
        double sx = straight_data.node_coords[2 * i];
        double sy = straight_data.node_coords[2 * i + 1];
        double cx = curved_data_identity.node_coords[2 * i];
        double cy = curved_data_identity.node_coords[2 * i + 1];
        assert(std::abs(sx - cx) < 1e-10 && "Identity mapping x should not change");
        assert(std::abs(sy - cy) < 1e-10 && "Identity mapping y should not change");
    }
    std::cout << "  Identity mapping verified!" << std::endl;

    // 测试3: 使用正弦扰动映射生成曲边网格
    std::cout << "\n=== Test 3: Generate curved mesh with sine perturbation (eps = 0.05) ===" << std::endl;
    Darcy::SinPerturbationMapping sin_map(0.05);
    success = generator.generate_from_straight(curved_base, sin_map);
    assert(success && "Failed to generate curved mesh with sine perturbation");

    const vem::CurvedMeshData& curved_data_sin = generator.get_mesh_data();
    std::cout << "  Curved mesh (sine): " << curved_data_sin.num_nodes << " nodes" << std::endl;

    // 打印曲边网格的节点坐标
    std::cout << "  Curved mesh nodes (first 10):" << std::endl;
    for (int i = 0; i < std::min(10, curved_data_sin.num_nodes); ++i) {
        double ref_x = straight_data.node_coords[2 * i];
        double ref_y = straight_data.node_coords[2 * i + 1];
        double phys_x = curved_data_sin.node_coords[2 * i];
        double phys_y = curved_data_sin.node_coords[2 * i + 1];
        std::cout << "    Node " << i << ": ref(" << ref_x << "," << ref_y
                  << ") -> phys(" << phys_x << "," << phys_y << ")" << std::endl;
    }

    // 验证坐标变化（扰动幅值约 0.05）
    std::cout << "  Verifying perturbation..." << std::endl;
    for (int i = 0; i < curved_data_sin.num_nodes; ++i) {
        double ref_x = straight_data.node_coords[2 * i];
        double ref_y = straight_data.node_coords[2 * i + 1];
        double phys_x = curved_data_sin.node_coords[2 * i];
        double phys_y = curved_data_sin.node_coords[2 * i + 1];

        // 手动计算期望的坐标
        double expected_x = ref_x + 0.05 * std::sin(2.0 * M_PI * ref_y);
        double expected_y = ref_y + 0.05 * std::sin(2.0 * M_PI * ref_x);

        assert(std::abs(phys_x - expected_x) < 1e-10 && "X coordinate mismatch");
        assert(std::abs(phys_y - expected_y) < 1e-10 && "Y coordinate mismatch");
    }
    std::cout << "  Sine perturbation verified!" << std::endl;

    // 测试4: 检查曲边网格的边界范围
    std::cout << "\n=== Test 4: Check curved mesh bounds ===" << std::endl;
    std::cout << "  Curved mesh bounds: x=[" << curved_data_sin.x_min << ", " << curved_data_sin.x_max
              << "], y=[" << curved_data_sin.y_min << ", " << curved_data_sin.y_max << "]" << std::endl;

    // 验证所有节点都在边界范围内
    for (int i = 0; i < curved_data_sin.num_nodes; ++i) {
        double x = curved_data_sin.node_coords[2 * i];
        double y = curved_data_sin.node_coords[2 * i + 1];
        assert(x >= curved_data_sin.x_min - 1e-10 && x <= curved_data_sin.x_max + 1e-10);
        assert(y >= curved_data_sin.y_min - 1e-10 && y <= curved_data_sin.y_max + 1e-10);
    }

    // 测试5: 更大的网格（square_2x2）
    std::cout << "\n=== Test 5: Curved mesh on square_2x2.msh ===" << std::endl;
    vem::StraightMeshReader straight_reader2;
    success = straight_reader2.read_mesh("/home/lzccode/code/vem/gmesh_py/mesh_data/square_2x2.msh");
    assert(success && "Failed to read square_2x2.msh");

    vem::CurvedMeshData curved_base2;
    straight_reader2.to_curved_mesh_data(curved_base2);

    success = generator.generate_from_straight(curved_base2, sin_map);
    assert(success && "Failed to generate curved mesh on square_2x2");

    const vem::CurvedMeshData& curved_data2 = generator.get_mesh_data();
    std::cout << "  Curved mesh 2x2: " << curved_data2.num_nodes << " nodes, "
              << curved_data2.num_cells << " cells" << std::endl;

    // 验证拓扑信息正确复制
    assert(curved_data2.num_nodes == curved_base2.num_nodes);
    assert(curved_data2.num_cells == curved_base2.num_cells);
    assert(curved_data2.num_edges == curved_base2.num_edges);
    assert(curved_data2.num_boundary_nodes == curved_base2.num_boundary_nodes);
    assert(curved_data2.num_boundary_edges == curved_base2.num_boundary_edges);
    std::cout << "  Topology correctly copied!" << std::endl;

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
