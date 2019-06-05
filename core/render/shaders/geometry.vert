#version 450 core

//The vertex might be reference or live, depends on the vao that binding to
layout(location = 0) in vec4 vertex_confid;
layout(location = 1) in vec4 normal_radius;
layout(location = 2) in vec4 color_time;

//The output used by normal map, phone shading and albedo color
out VertexOut {
    vec4 camera_vertex;
    vec4 camera_normal;
    vec4 normalized_rgb;
} vertex_shader_out;

uniform mat4 world2camera;
uniform vec4 intrinsic; //cx, cy, fx, fy
//The width and height are in pixels, the maxdepth is in [m]
//The last element is current time
uniform vec4 width_height_maxdepth_currtime;
uniform vec2 confid_time_threshold; // x is confidence threshold, y is time threshold

vec3 project_point(vec3 p)
{
    return vec3(( (((intrinsic.z * p.x) / p.z) + intrinsic.x) - (width_height_maxdepth_currtime.x * 0.5)) / (width_height_maxdepth_currtime.x * 0.5),
                ((((intrinsic.w * p.y) / p.z) + intrinsic.y) - (width_height_maxdepth_currtime.y * 0.5)) / (width_height_maxdepth_currtime.y * 0.5),
                p.z / (width_height_maxdepth_currtime.z + 0.05));
}

vec3 project_point_image(vec3 p)
{
    return vec3(((intrinsic.z * p.x) / p.z) + intrinsic.x,
                ((intrinsic.w * p.y) / p.z) + intrinsic.y,
                p.z);
}

void main() {
    vec4 vertex_camera = world2camera * vec4(vertex_confid.xyz, 1.0);
    if(  vertex_camera.z > (width_height_maxdepth_currtime.z + 0.05)
      || vertex_camera.z < 0
      || (vertex_confid.w < confid_time_threshold.x && (abs(color_time.z - width_height_maxdepth_currtime.w) >= confid_time_threshold.y))
    ) {
        //Make it outside the screes of [-1 1]
        gl_Position = vec4(1000.0f, 1000.0f, 1000.0f, 1000.0f);
    } else {
        //Should be inside the range
        gl_Position = vec4(project_point(vertex_camera.xyz), 1.0);

        //Decode the color
        int encoded_color = floatBitsToInt(color_time.x);
        vec3 color;
        color.x = float(((encoded_color & 0x00FF0000) >> 16)) / 255.f;
        color.y = float(((encoded_color & 0x0000FF00) >>  8)) / 255.f;
        color.z = float(  encoded_color & 0x000000FF) / 255.f;

        //Compute the size of the surfel
        mat3 world2camera_R = mat3(world2camera);
        vec4 normRad = vec4(normalize(world2camera_R * normal_radius.xyz), normal_radius.w);

        //Note that radius is in mm
        vec3 x1 = normalize(vec3((normRad.y - normRad.z), -normRad.x, normRad.x)) * normal_radius.w * 1.41421356 * 0.001;
	    vec3 y1 = cross(normRad.xyz, x1);
        vec4 proj1 = vec4(project_point_image(vertex_camera.xyz + x1), 1.0);
	    vec4 proj2 = vec4(project_point_image(vertex_camera.xyz + y1), 1.0);
	    vec4 proj3 = vec4(project_point_image(vertex_camera.xyz - y1), 1.0);
	    vec4 proj4 = vec4(project_point_image(vertex_camera.xyz - x1), 1.0);
        vec2 xs = vec2(min(proj1.x, min(proj2.x, min(proj3.x, proj4.x))), max(proj1.x, max(proj2.x, max(proj3.x, proj4.x))));
	    vec2 ys = vec2(min(proj1.y, min(proj2.y, min(proj3.y, proj4.y))), max(proj1.y, max(proj2.y, max(proj3.y, proj4.y))));

        //Please refer to Elastic Fusion for these codes
        float xDiff = abs(xs.y - xs.x);
	    float yDiff = abs(ys.y - ys.x);
	    gl_PointSize = max(0.0f, min(xDiff, yDiff));

        //Fill in the output
        vertex_shader_out.camera_vertex = vec4(vertex_camera.xyz, 1.0);
        vertex_shader_out.camera_normal = vec4(normRad.xyz, 1.0);
        vertex_shader_out.normalized_rgb = vec4(color, 1.0);
    }
}
