#version 450 core

flat in int vs_out_vertex_id;
in vec4 vs_out_warp_vertex;
in vec4 vs_out_warp_normal;
flat in vec4 vs_out_color_time;

layout(location = 0) out vec4 warp_vertex_map;
layout(location = 1) out vec4 warp_normal_map;
layout(location = 2) out int index_map;
layout(location = 3) out vec4 color_time_map;


void main() {
    index_map = vs_out_vertex_id;
    warp_vertex_map = vs_out_warp_vertex;
    warp_normal_map = vs_out_warp_normal;
    color_time_map = vs_out_color_time;
}