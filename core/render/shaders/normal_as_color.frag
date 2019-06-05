#version 450 core

in VertexOut {
    vec4 camera_vertex;
    vec4 camera_normal;
    vec4 normalized_rgb;
} fs_in;

out vec4 fragment_color;

void main() {
    vec3 normal = normalize(fs_in.camera_normal.xyz);
    fragment_color = vec4(normal * 0.5 + 0.5, 1.0);
}