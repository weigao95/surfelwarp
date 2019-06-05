#version 450 core

in vec4 vs_out_reference_vertex;
in vec4 vs_out_reference_normal;
in vec4 vs_out_warp_vertex;
in vec4 vs_out_warp_normal;
flat in int vs_out_vertex_id;
in vec4 vs_out_normalized_rgb;

layout(location = 0) out vec4 reference_vertex_map;
layout(location = 1) out vec4 reference_normal_map;
layout(location = 2) out vec4 warp_vertex_map;
layout(location = 3) out vec4 warp_normal_map;
layout(location = 4) out int index_map;
layout(location = 5) out vec4 normalized_rgb_map;

void main() {
    reference_vertex_map = vs_out_reference_vertex;
    reference_normal_map = vs_out_reference_normal;
    warp_vertex_map = vs_out_warp_vertex;
    warp_normal_map = vs_out_warp_normal;
    index_map = vs_out_vertex_id;
    normalized_rgb_map = vs_out_normalized_rgb;
}