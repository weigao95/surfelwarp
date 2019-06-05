#version 450 core

layout(location = 0) in vec4 warp_vertex;
layout(location = 1) in vec4 warp_normal;
layout(location = 2) in vec4 color_time;


out vec4 vs_out_warp_vertex;
out vec4 vs_out_warp_normal;
flat out int vs_out_vertex_id;
flat out vec4 vs_out_color_time;


uniform mat4 world2camera;
uniform vec4 intrinsic; //cx, cy, fx, fy
//The width and height are in pixels, the maxdepth is in [m]
//The last element is not used
uniform vec4 width_height_maxdepth; 

vec3 project_point(vec3 p)
{
    return vec3(( (((intrinsic.z * p.x) / p.z) + intrinsic.x) - (width_height_maxdepth.x * 0.5)) / (width_height_maxdepth.x * 0.5),
                ((((intrinsic.w * p.y) / p.z) + intrinsic.y) - (width_height_maxdepth.y * 0.5)) / (width_height_maxdepth.y * 0.5),
                p.z / (width_height_maxdepth.z + 0.05));
}

void main() {
    vec4 warp_vertex_camera = world2camera * vec4(warp_vertex.xyz, 1.0);
    if(warp_vertex_camera.z > (width_height_maxdepth.z + 0.05)
    || warp_vertex_camera.z < 0
    ) {
        //Make it outside the screes of [-1 1]
        gl_Position = vec4(1000.0f, 1000.0f, 1000.0f, 1000.0f);
    }
    else {
        gl_Position = vec4(project_point(warp_vertex_camera.xyz), 1.0);

        // Collect information for fragment shader input
        vs_out_warp_vertex =  warp_vertex;
        vs_out_warp_normal = warp_normal;
        vs_out_vertex_id = gl_VertexID;
        vs_out_color_time = color_time;
    }
}