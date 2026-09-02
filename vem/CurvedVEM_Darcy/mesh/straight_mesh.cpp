#include "straight_mesh.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace vem {

bool StraightMeshReader::read_mesh(const std::string& filename) {
    mesh_data_.clear();

    if (!read_coords(filename)) {
        std::cerr << "Error: Failed to read mesh coordinates from " << filename << std::endl;
        return false;
    }

    if (!read_elements(filename)) {
        std::cerr << "Error: Failed to read mesh elements from " << filename << std::endl;
        return false;
    }

    if (!compute_cell_properties()) {
        std::cerr << "Error: Failed to compute cell properties" << std::endl;
        return false;
    }

    if (!generate_element_edges()) {
        std::cerr << "Error: Failed to generate element edges" << std::endl;
        return false;
    }

    if (!compute_boundary_info()) {
        std::cerr << "Error: Failed to compute boundary info" << std::endl;
        return false;
    }

    std::cout << "Successfully read mesh: " << mesh_data_.num_nodes << " nodes, "
              << mesh_data_.num_cells << " cells, " << mesh_data_.num_edges << " edges." << std::endl;
    return true;
}

bool StraightMeshReader::read_coords(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    bool found_nodes = false;
    while (std::getline(file, line)) {
        if (line.find("$Nodes") != std::string::npos) {
            found_nodes = true;
            break;
        }
    }

    if (!found_nodes) {
        std::cerr << "Error: $Nodes section not found in file!" << std::endl;
        return false;
    }

    if (!std::getline(file, line)) {
        std::cerr << "Error: Failed to read nodes header!" << std::endl;
        return false;
    }

    std::istringstream header_stream(line);
    int num_entity_blocks, num_nodes;
    header_stream >> num_entity_blocks >> num_nodes;

    if (num_nodes <= 0) {
        std::cerr << "Error: Invalid number of nodes!" << std::endl;
        return false;
    }

    mesh_data_.num_nodes = num_nodes;
    mesh_data_.node_coords.assign(2 * num_nodes, 0.0);

    int nodes_read = 0;
    for (int block = 0; block < num_entity_blocks; ++block) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Failed to read entity block " << block << std::endl;
            return false;
        }

        std::istringstream block_stream(line);
        int entity_dim, entity_tag, parametric, num_nodes_in_block;
        block_stream >> entity_dim >> entity_tag >> parametric >> num_nodes_in_block;

        std::vector<int> node_tags;
        node_tags.reserve(num_nodes_in_block);
        int tags_read = 0;
        while (tags_read < num_nodes_in_block) {
            if (!std::getline(file, line)) {
                std::cerr << "Error: Failed to read node tags in block " << block << std::endl;
                return false;
            }
            std::istringstream tag_stream(line);
            int tag;
            while (tag_stream >> tag && tags_read < num_nodes_in_block) {
                node_tags.push_back(tag);
                tags_read++;
            }
        }

        for (int i = 0; i < num_nodes_in_block; ++i) {
            if (!std::getline(file, line)) {
                std::cerr << "Error: Failed to read node coordinates in block " << block << std::endl;
                return false;
            }

            std::istringstream coord_stream(line);
            double x, y, z;
            if (!(coord_stream >> x >> y >> z)) {
                continue;
            }

            int node_id = node_tags[i] - 1; // 转换为0-based
            if (node_id >= 0 && node_id < num_nodes) {
                mesh_data_.node_coords[2 * node_id] = x;
                mesh_data_.node_coords[2 * node_id + 1] = y;
                nodes_read++;
            }
        }
    }

    return true;
}

bool StraightMeshReader::read_elements(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    bool found_elements = false;
    while (std::getline(file, line)) {
        if (line.find("$Elements") != std::string::npos) {
            found_elements = true;
            break;
        }
    }

    if (!found_elements) {
        std::cerr << "Error: $Elements section not found in file!" << std::endl;
        return false;
    }

    if (!std::getline(file, line)) {
        std::cerr << "Error: Failed to read elements header!" << std::endl;
        return false;
    }

    std::istringstream header_stream(line);
    int num_entity_blocks, num_elements;
    header_stream >> num_entity_blocks >> num_elements;

    std::vector<int> temp_nodes_per_cell;
    int total_vertices = 0;
    int poly_element_count = 0;

    std::streampos current_pos = file.tellg();
    for (int block = 0; block < num_entity_blocks; ++block) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Failed to read entity block " << block << std::endl;
            return false;
        }

        std::istringstream block_stream(line);
        int entity_dim, entity_tag, element_type, num_elements_in_block;
        block_stream >> entity_dim >> entity_tag >> element_type >> num_elements_in_block;

        int nodes_per_elem = get_nodes_count_by_element_type(element_type);
        if (nodes_per_elem < 3) {
            for (int i = 0; i < num_elements_in_block; ++i) {
                std::getline(file, line);
            }
            continue;
        }

        for (int i = 0; i < num_elements_in_block; ++i) {
            std::getline(file, line);
            temp_nodes_per_cell.push_back(nodes_per_elem);
            total_vertices += nodes_per_elem;
            poly_element_count++;
        }
    }

    mesh_data_.num_cells = poly_element_count;
    if (mesh_data_.num_cells == 0) {
        std::cerr << "Error: No valid polygon elements found!" << std::endl;
        return false;
    }

    mesh_data_.nodes_per_cell = temp_nodes_per_cell;
    mesh_data_.cell_nodes.reserve(total_vertices);
    mesh_data_.cell_node_indices.push_back(0);
    for (int i = 1; i <= mesh_data_.num_cells; ++i) {
        mesh_data_.cell_node_indices.push_back(
            mesh_data_.cell_node_indices[i-1] + mesh_data_.nodes_per_cell[i-1]
        );
    }

    file.clear();
    file.seekg(0);
    found_elements = false;
    while (std::getline(file, line)) {
        if (line.find("$Elements") != std::string::npos) {
            found_elements = true;
            break;
        }
    }

    if (!found_elements) {
        std::cerr << "Error: $Elements section not found again!" << std::endl;
        return false;
    }

    std::getline(file, line);
    int elem_idx = 0;

    for (int block = 0; block < num_entity_blocks; ++block) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Failed to read entity block " << block << std::endl;
            return false;
        }

        std::istringstream block_stream(line);
        int entity_dim, entity_tag, element_type, num_elements_in_block;
        block_stream >> entity_dim >> entity_tag >> element_type >> num_elements_in_block;

        int nodes_per_elem = get_nodes_count_by_element_type(element_type);
        if (nodes_per_elem < 3) {
            for (int i = 0; i < num_elements_in_block; ++i) {
                std::getline(file, line);
            }
            continue;
        }

        for (int i = 0; i < num_elements_in_block; ++i) {
            if (!std::getline(file, line)) {
                std::cerr << "Error: Failed to read element " << i << " in block " << block << std::endl;
                return false;
            }

            std::istringstream elem_stream(line);
            int elem_tag;
            elem_stream >> elem_tag;

            std::vector<int> nodes(nodes_per_elem);
            for (int j = 0; j < nodes_per_elem; ++j) {
                elem_stream >> nodes[j];
                nodes[j] -= 1; // 转换为0-based
            }

            reorder_polygon_to_ccw(nodes, mesh_data_.node_coords);
            mesh_data_.cell_nodes.insert(mesh_data_.cell_nodes.end(), nodes.begin(), nodes.end());
            elem_idx++;
        }
    }

    compute_mesh_bounds();
    return true;
}

int StraightMeshReader::get_nodes_count_by_element_type(int element_type) {
    switch (element_type) {
        case 2:  return 3;   // 三角形
        case 3:  return 4;   // 四边形
        case 11: return 5;   // 五边形
        case 12: return 6;   // 六边形
        default: return -1;  // 不支持的类型
    }
}

double StraightMeshReader::polygon_signed_area(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    double area = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += (x[i] * y[j]) - (x[j] * y[i]);
    }
    return area / 2.0;
}

void StraightMeshReader::reorder_polygon_to_ccw(std::vector<int>& nodes, const std::vector<double>& global_node_coords) {
    int n = nodes.size();
    if (n < 3) return;

    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        int node_id = nodes[i];
        if (node_id >= 0 && 2 * node_id + 1 < static_cast<int>(global_node_coords.size())) {
            x[i] = global_node_coords[2 * node_id];
            y[i] = global_node_coords[2 * node_id + 1];
        } else {
            x[i] = 0.0;
            y[i] = 0.0;
        }
    }

    double area = polygon_signed_area(x, y);
    if (area < 0) {
        std::reverse(nodes.begin(), nodes.end());
    }
}

void StraightMeshReader::compute_mesh_bounds() {
    if (mesh_data_.node_coords.empty() || mesh_data_.num_nodes == 0) {
        return;
    }

    mesh_data_.x_min = mesh_data_.node_coords[0];
    mesh_data_.x_max = mesh_data_.node_coords[0];
    mesh_data_.y_min = mesh_data_.node_coords[1];
    mesh_data_.y_max = mesh_data_.node_coords[1];

    for (int i = 0; i < mesh_data_.num_nodes; ++i) {
        double x = mesh_data_.node_coords[2 * i];
        double y = mesh_data_.node_coords[2 * i + 1];
        if (x < mesh_data_.x_min) mesh_data_.x_min = x;
        if (x > mesh_data_.x_max) mesh_data_.x_max = x;
        if (y < mesh_data_.y_min) mesh_data_.y_min = y;
        if (y > mesh_data_.y_max) mesh_data_.y_max = y;
    }
}

bool StraightMeshReader::compute_cell_properties() {
    if (mesh_data_.node_coords.empty() || mesh_data_.num_cells == 0 ||
        mesh_data_.cell_nodes.empty() || mesh_data_.cell_node_indices.empty()) {
        std::cerr << "Error: No cell/node data available!" << std::endl;
        return false;
    }

    mesh_data_.cell_centroid_x.resize(mesh_data_.num_cells, 0.0);
    mesh_data_.cell_centroid_y.resize(mesh_data_.num_cells, 0.0);
    mesh_data_.cell_area.resize(mesh_data_.num_cells, 0.0);
    mesh_data_.cell_diameter.resize(mesh_data_.num_cells, 0.0);

    for (int cell_id = 0; cell_id < mesh_data_.num_cells; ++cell_id) {
        int start_idx = mesh_data_.cell_node_indices[cell_id];
        int node_count = mesh_data_.nodes_per_cell[cell_id];

        std::vector<double> x(node_count), y(node_count);
        for (int i = 0; i < node_count; ++i) {
            int global_node_id = mesh_data_.cell_nodes[start_idx + i];
            if (global_node_id >= 0 && global_node_id < mesh_data_.num_nodes) {
                x[i] = mesh_data_.node_coords[2 * global_node_id];
                y[i] = mesh_data_.node_coords[2 * global_node_id + 1];
            } else {
                x[i] = 0.0;
                y[i] = 0.0;
            }
        }

        if (node_count == 4) {
            double x_min = x[0], x_max = x[0];
            double y_min = y[0], y_max = y[0];
            for (int i = 1; i < 4; ++i) {
                x_min = std::min(x_min, x[i]);
                x_max = std::max(x_max, x[i]);
                y_min = std::min(y_min, y[i]);
                y_max = std::max(y_max, y[i]);
            }
            mesh_data_.cell_centroid_x[cell_id] = (x_min + x_max) / 2.0;
            mesh_data_.cell_centroid_y[cell_id] = (y_min + y_max) / 2.0;
            mesh_data_.cell_area[cell_id] = (x_max - x_min) * (y_max - y_min);
        } else {
            double cx_sum = 0.0, cy_sum = 0.0;
            for (int i = 0; i < node_count; ++i) {
                cx_sum += x[i];
                cy_sum += y[i];
            }
            mesh_data_.cell_centroid_x[cell_id] = cx_sum / node_count;
            mesh_data_.cell_centroid_y[cell_id] = cy_sum / node_count;

            double area = 0.0;
            for (int i = 0; i < node_count; ++i) {
                int j = (i + 1) % node_count;
                area += x[i] * y[j] - x[j] * y[i];
            }
            mesh_data_.cell_area[cell_id] = fabs(area) / 2.0;
        }

        double max_dist = 0.0;
        for (int i = 0; i < node_count; ++i) {
            for (int j = i + 1; j < node_count; ++j) {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dist = sqrt(dx * dx + dy * dy);
                if (dist > max_dist) {
                    max_dist = dist;
                }
            }
        }
        mesh_data_.cell_diameter[cell_id] = max_dist;
    }

    return true;
}

int StraightMeshReader::edge_hash_key(int a, int b) const {
    int min_node = std::min(a, b);
    int max_node = std::max(a, b);
    return (min_node << 16) | (max_node & 0xFFFF);
}

bool StraightMeshReader::generate_element_edges() {
    if (mesh_data_.cell_nodes.empty() || mesh_data_.num_cells == 0) {
        std::cerr << "Error: No cell/node data available!" << std::endl;
        return false;
    }

    std::vector<std::pair<int, int>> all_edges_with_duplicates;
    std::vector<std::pair<int, int>> elem_edges_original;

    for (int c = 0; c < mesh_data_.num_cells; ++c) {
        int elem_node_start = mesh_data_.cell_node_indices[c];
        int n_edges = mesh_data_.nodes_per_cell[c];

        for (int local_edge = 0; local_edge < n_edges; ++local_edge) {
            int node1_idx = elem_node_start + local_edge;
            int node2_idx = elem_node_start + ((local_edge + 1) % n_edges);
            int node_a = mesh_data_.cell_nodes[node1_idx];
            int node_b = mesh_data_.cell_nodes[node2_idx];

            int min_node = std::min(node_a, node_b);
            int max_node = std::max(node_a, node_b);
            auto edge = std::make_pair(min_node, max_node);
            all_edges_with_duplicates.push_back(edge);
            elem_edges_original.push_back(edge);
        }
    }

    std::vector<std::pair<std::pair<int, int>, int>> edge_with_index;
    edge_with_index.reserve(all_edges_with_duplicates.size());
    for (int i = 0; i < static_cast<int>(all_edges_with_duplicates.size()); ++i) {
        edge_with_index.emplace_back(all_edges_with_duplicates[i], i);
    }

    std::sort(edge_with_index.begin(), edge_with_index.end());

    std::vector<int> global_edge_id_map(all_edges_with_duplicates.size(), -1);
    std::vector<std::pair<int, int>> unique_edges;
    int global_edge_count = 0;

    for (int i = 0; i < static_cast<int>(edge_with_index.size()); ++i) {
        if (i == 0 || edge_with_index[i].first != edge_with_index[i-1].first) {
            unique_edges.push_back(edge_with_index[i].first);
            global_edge_count++;
        }
        global_edge_id_map[edge_with_index[i].second] = global_edge_count - 1;
    }

    std::vector<int> edge_occurrence(global_edge_count, 0);
    for (int i = 0; i < static_cast<int>(elem_edges_original.size()); ++i) {
        int edge_id = global_edge_id_map[i];
        edge_occurrence[edge_id]++;
    }

    std::vector<int> elem_edges_temp;
    elem_edges_temp.reserve(elem_edges_original.size());
    for (int i = 0; i < static_cast<int>(elem_edges_original.size()); ++i) {
        elem_edges_temp.push_back(global_edge_id_map[i]);
    }

    mesh_data_.cell_edge_indices.resize(mesh_data_.num_cells + 1, 0);
    int current_edge_idx = 0;
    for (int c = 0; c < mesh_data_.num_cells; ++c) {
        current_edge_idx += mesh_data_.nodes_per_cell[c];
        mesh_data_.cell_edge_indices[c+1] = current_edge_idx;
    }

    mesh_data_.num_edges = global_edge_count;
    mesh_data_.global_edges = std::move(elem_edges_temp);
    mesh_data_.edge_occurrence = std::move(edge_occurrence);
    mesh_data_.edge_endpoints.resize(mesh_data_.num_edges * 2);
    for (int i = 0; i < mesh_data_.num_edges; ++i) {
        mesh_data_.edge_endpoints[2 * i] = unique_edges[i].first;
        mesh_data_.edge_endpoints[2 * i + 1] = unique_edges[i].second;
    }

    return true;
}

bool StraightMeshReader::compute_boundary_info() {
    if (mesh_data_.node_coords.empty() || mesh_data_.num_nodes == 0 ||
        mesh_data_.edge_occurrence.empty()) {
        std::cerr << "Error: Invalid mesh data for boundary computation!" << std::endl;
        return false;
    }

    const double tol = 1e-6;

    std::vector<int> temp_boundary_nodes;
    for (int i = 0; i < mesh_data_.num_nodes; ++i) {
        double x = mesh_data_.node_coords[2 * i];
        double y = mesh_data_.node_coords[2 * i + 1];
        if (fabs(x - mesh_data_.x_min) < tol || fabs(x - mesh_data_.x_max) < tol ||
            fabs(y - mesh_data_.y_min) < tol || fabs(y - mesh_data_.y_max) < tol) {
            temp_boundary_nodes.push_back(i);
        }
    }

    std::vector<int> bottom_nodes, right_nodes, top_nodes, left_nodes;
    for (int node_id : temp_boundary_nodes) {
        double x = mesh_data_.node_coords[2 * node_id];
        double y = mesh_data_.node_coords[2 * node_id + 1];

        if (fabs(y - mesh_data_.y_min) < tol && !(fabs(x - mesh_data_.x_max) < tol)) {
            bottom_nodes.push_back(node_id);
        } else if (fabs(x - mesh_data_.x_max) < tol && !(fabs(y - mesh_data_.y_max) < tol)) {
            right_nodes.push_back(node_id);
        } else if (fabs(y - mesh_data_.y_max) < tol && !(fabs(x - mesh_data_.x_min) < tol)) {
            top_nodes.push_back(node_id);
        } else if (fabs(x - mesh_data_.x_min) < tol) {
            left_nodes.push_back(node_id);
        }
    }

    std::sort(bottom_nodes.begin(), bottom_nodes.end(), [this](int a, int b) {
        return mesh_data_.node_coords[2 * a] < mesh_data_.node_coords[2 * b];
    });
    std::sort(right_nodes.begin(), right_nodes.end(), [this](int a, int b) {
        return mesh_data_.node_coords[2 * a + 1] < mesh_data_.node_coords[2 * b + 1];
    });
    std::sort(top_nodes.begin(), top_nodes.end(), [this](int a, int b) {
        return mesh_data_.node_coords[2 * a] > mesh_data_.node_coords[2 * b];
    });
    std::sort(left_nodes.begin(), left_nodes.end(), [this](int a, int b) {
        return mesh_data_.node_coords[2 * a + 1] > mesh_data_.node_coords[2 * b + 1];
    });

    std::vector<int> boundary_nodes_temp;
    boundary_nodes_temp.insert(boundary_nodes_temp.end(), bottom_nodes.begin(), bottom_nodes.end());
    boundary_nodes_temp.insert(boundary_nodes_temp.end(), right_nodes.begin(), right_nodes.end());
    boundary_nodes_temp.insert(boundary_nodes_temp.end(), top_nodes.begin(), top_nodes.end());
    boundary_nodes_temp.insert(boundary_nodes_temp.end(), left_nodes.begin(), left_nodes.end());
    boundary_nodes_temp.push_back(boundary_nodes_temp.front());

    int boundary_edge_count = static_cast<int>(boundary_nodes_temp.size() - 1);
    std::vector<int> boundary_edges_temp;
    boundary_edges_temp.reserve(boundary_edge_count);

    for (int i = 0; i < boundary_edge_count; ++i) {
        int node1 = boundary_nodes_temp[i];
        int node2 = boundary_nodes_temp[i + 1];
        int edge_id = -1;

        for (int j = 0; j < mesh_data_.num_edges; ++j) {
            int e1 = mesh_data_.edge_endpoints[2 * j];
            int e2 = mesh_data_.edge_endpoints[2 * j + 1];
            if ((e1 == node1 && e2 == node2) || (e1 == node2 && e2 == node1)) {
                edge_id = j;
                break;
            }
        }
        boundary_edges_temp.push_back(edge_id);
    }

    int idx_bottom_start = 0;
    int idx_right_start = static_cast<int>(bottom_nodes.size());
    int idx_top_start = static_cast<int>(bottom_nodes.size() + right_nodes.size());
    int idx_left_start = static_cast<int>(bottom_nodes.size() + right_nodes.size() + top_nodes.size());

    mesh_data_.num_boundary_nodes = static_cast<int>(boundary_nodes_temp.size());
    mesh_data_.boundary_nodes = std::move(boundary_nodes_temp);
    mesh_data_.bnode_indices = {idx_bottom_start, idx_right_start, idx_top_start, idx_left_start, mesh_data_.num_boundary_nodes};

    mesh_data_.num_boundary_edges = boundary_edge_count;
    mesh_data_.boundary_edges = std::move(boundary_edges_temp);
    mesh_data_.bedge_indices = {idx_bottom_start, idx_right_start, idx_top_start, idx_left_start, boundary_edge_count};

    return true;
}

bool StraightMeshReader::validate_input(int cell_id, int local_edge_id) const {
    if (cell_id < 0 || cell_id >= mesh_data_.num_cells) {
        std::cerr << "Error: Cell ID " << cell_id << " out of range (0~"
                  << mesh_data_.num_cells - 1 << ")" << std::endl;
        return false;
    }

    int edge_count = mesh_data_.nodes_per_cell[cell_id];
    if (local_edge_id < 0 || local_edge_id >= edge_count) {
        std::cerr << "Error: Local edge ID " << local_edge_id
                  << " out of range (0~" << edge_count - 1 << ")" << std::endl;
        return false;
    }

    if (mesh_data_.cell_edge_indices.empty() || mesh_data_.global_edges.empty()) {
        std::cerr << "Error: Edge data not initialized!" << std::endl;
        return false;
    }

    return true;
}

int StraightMeshReader::find_adjacent_cell(int cell_id, int local_edge_id) const {
    if (!validate_input(cell_id, local_edge_id)) {
        return -1;
    }

    int edge_start_idx = mesh_data_.cell_edge_indices[cell_id];
    int global_edge_id = mesh_data_.global_edges[edge_start_idx + local_edge_id];

    int adjacent_cell_id = cell_id;
    for (int elem_idx = 0; elem_idx < mesh_data_.num_cells; ++elem_idx) {
        if (elem_idx == cell_id) continue;

        int curr_edge_start = mesh_data_.cell_edge_indices[elem_idx];
        int curr_edge_count = mesh_data_.nodes_per_cell[elem_idx];

        for (int local_idx = 0; local_idx < curr_edge_count; ++local_idx) {
            int curr_global_edge = mesh_data_.global_edges[curr_edge_start + local_idx];
            if (curr_global_edge == global_edge_id) {
                adjacent_cell_id = elem_idx;
                goto END_SEARCH;
            }
        }
    }

END_SEARCH:
    if (mesh_data_.edge_occurrence.size() > static_cast<size_t>(global_edge_id)) {
        if (mesh_data_.edge_occurrence[global_edge_id] == 1) {
            adjacent_cell_id = cell_id;
        }
    }

    return adjacent_cell_id;
}

std::pair<int, int> StraightMeshReader::find_adjacent_cells_by_edge(int global_edge_id) const {
    if (global_edge_id < 0 || global_edge_id >= mesh_data_.num_edges) {
        std::cerr << "Error: Global edge ID " << global_edge_id << " out of range (0~"
                  << mesh_data_.num_edges - 1 << ")" << std::endl;
        return {-1, -1};
    }

    if (mesh_data_.cell_edge_indices.empty() || mesh_data_.global_edges.empty()) {
        std::cerr << "Error: Edge data not initialized!" << std::endl;
        return {-1, -1};
    }

    std::vector<int> found_cells;
    for (int elem_idx = 0; elem_idx < mesh_data_.num_cells; ++elem_idx) {
        int curr_edge_start = mesh_data_.cell_edge_indices[elem_idx];
        int curr_edge_count = mesh_data_.nodes_per_cell[elem_idx];

        for (int local_idx = 0; local_idx < curr_edge_count; ++local_idx) {
            int curr_global_edge = mesh_data_.global_edges[curr_edge_start + local_idx];
            if (curr_global_edge == global_edge_id) {
                found_cells.push_back(elem_idx);
                if (found_cells.size() >= 2) {
                    goto END_SEARCH;
                }
            }
        }
    }

END_SEARCH:
    if (found_cells.size() == 2) {
        return {found_cells[0], found_cells[1]};
    } else if (found_cells.size() == 1) {
        return {found_cells[0], found_cells[0]};
    } else {
        std::cerr << "Error: No adjacent cells found for edge " << global_edge_id << std::endl;
        return {-1, -1};
    }
}

void StraightMeshReader::to_curved_mesh_data(CurvedMeshData& out) const {
    out.clear();

    // 复制节点信息
    out.num_nodes = mesh_data_.num_nodes;
    out.node_coords = mesh_data_.node_coords;

    // 复制单元信息
    out.num_cells = mesh_data_.num_cells;
    out.cell_nodes = mesh_data_.cell_nodes;
    out.cell_node_indices = mesh_data_.cell_node_indices;
    out.nodes_per_cell = mesh_data_.nodes_per_cell;

    // 复制边信息
    out.num_edges = mesh_data_.num_edges;
    out.cell_edge_indices = mesh_data_.cell_edge_indices;
    out.global_edges = mesh_data_.global_edges;
    out.edge_endpoints = mesh_data_.edge_endpoints;
    out.edge_occurrence = mesh_data_.edge_occurrence;

    // 复制边界信息
    out.num_boundary_nodes = mesh_data_.num_boundary_nodes;
    out.boundary_nodes = mesh_data_.boundary_nodes;
    out.bnode_indices = mesh_data_.bnode_indices;
    out.num_boundary_edges = mesh_data_.num_boundary_edges;
    out.boundary_edges = mesh_data_.boundary_edges;
    out.bedge_indices = mesh_data_.bedge_indices;

    // 复制边界范围
    out.x_min = mesh_data_.x_min;
    out.x_max = mesh_data_.x_max;
    out.y_min = mesh_data_.y_min;
    out.y_max = mesh_data_.y_max;
}

} // namespace vem
