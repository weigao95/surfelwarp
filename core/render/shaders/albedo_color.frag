#version 450 core

in VertexOut {
    vec4 camera_vertex;
    vec4 camera_normal;
    vec4 normalized_rgb;
} fs_in;

out vec4 fragment_color;

void main() {
    fragment_color = fs_in.normalized_rgb;
}