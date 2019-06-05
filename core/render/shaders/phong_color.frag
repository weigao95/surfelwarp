#version 450 core

in VertexOut {
    vec4 camera_vertex;
    vec4 camera_normal;
    vec4 normalized_rgb;
} fs_in;

out vec4 fragment_color;

void main() {
    vec3 ldir = normalize(-vec3(0.0, 0.0, 1.0)); //The light direction
    vec3 fn = normalize(fs_in.camera_normal.xyz);
    vec3 vdir = normalize(-fs_in.camera_vertex.xyz);
    vec3 rdir = reflect(-ldir, fn);

    //Discard invalid normal
    if(any(isnan(fn))) discard;

    //The pre-defined ambient, diffuse and specular
    vec3 ambient = vec3(0.3, 0.3, 0.3);
    vec3 diffuse = vec3(0.3, 0.3, 0.3);
    vec3 specular = vec3(0.5, 0.5, 0.5);

    //Compute the phong lighting for fragment
    vec3 color_a = ambient;
    vec3 color_d = diffuse * max(0.0f, dot(fn, ldir));
    vec3 color_s = specular * pow(max(dot(vdir, rdir), 0.0), 1.0);

    vec3 fc = clamp(color_a + color_d + color_s, 0.0, 1.0);
    fragment_color = vec4(fc, 1.0);
}