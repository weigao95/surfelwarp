//
// Created by wei on 3/18/18.
//
#include "core/render/Renderer.h"
#include "common/logging.h"
#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "Renderer.h"
#include "core/SurfelWarpSerial.h"

surfelwarp::Renderer::Renderer(int image_rows, int image_cols)
: m_image_width(image_cols)
, m_image_height(image_rows)
, m_fusion_map_width(image_cols * Constants::kFusionMapScale)
, m_fusion_map_height(image_rows * Constants::kFusionMapScale)
, m_io(ImGui::GetIO())
{
    if (!glfwInit()) {
        LOG(FATAL) << "The graphic pipeline is not correctly initialized";
    }

    // Assign the depth and height
    const auto& config = ConfigParser::Instance();
    m_renderer_intrinsic = config.rgb_intrinsic_clip();
    m_width_height_maxdepth = make_float4(
        image_cols, image_rows, config.max_rendering_depth(), 0.0f);

    // A series of sub-init functions
    initGLFW();
    initImGui();
    initClearValues();
    initVertexBufferObjects(); // The vertex buffer objects
    initMapRenderVAO(); // The vao, must after vbos
    initFrameRenderBuffers();
    initShaders();
}

surfelwarp::Renderer::~Renderer()
{
    // A series of sub-free functions
    freeVertexBufferObjects();
    freeFrameRenderBuffers();
}

/* GLFW window related functions
 */
void surfelwarp::Renderer::initGLFW()
{
    // The primary monitor
    mGLFWmonitor = glfwGetPrimaryMonitor();

    // The opengl context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Defualt framebuffer properties
    const auto& config = ConfigParser::Instance();
    if (!config.isShowOnline()) {
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    }
    glfwWindowHint(GLFW_SAMPLES, 1);
    glfwWindowHint(GLFW_STEREO, GL_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);

    // Switch to second montior
    int monitor_count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
    if (monitor_count > 1) {
        mGLFWmonitor = monitors[1];
    }

    // Setup of the window
    mGLFWwindow = glfwCreateWindow(1280, 720, "Non-rigid Recon", NULL, NULL);
    if (mGLFWwindow == NULL) {
        LOG(FATAL) << "The GLFW window is not correctly created";
    }

    // Make newly created context current
    glfwMakeContextCurrent(mGLFWwindow);
    // glfwSetFramebufferSizeCallback(mGLFWwindow, framebuffer_size_callback);

    // Init glad
    if (!gladLoadGL()) {
        LOG(FATAL) << "Glad is not correctly initialized";
    }

    // Enable depth test, disable face culling
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
}

void surfelwarp::Renderer::framebuffer_size_callback(
    GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void surfelwarp::Renderer::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mGLFWwindow, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void surfelwarp::Renderer::ImGuiWindow(int current_time)
{
    static float f = 0.0f;
    static int counter = 0;
    bool use_reinit = surfelwarp::SurfelWarpSerial::get_use_periodic_reinit();
    ImGui::SetNextWindowPos(ImVec2(1200, 720), ImGuiCond_FirstUseEver);

    ImGui::Begin("Surfelwarp!"); // Create a window called "Hello,
                                 // world!" and append into it.
    ImGui::Text("current_frame_index = %d", current_time);
    ImGui::Checkbox("Use reinit", &use_reinit);
    surfelwarp::SurfelWarpSerial::set_use_periodic_reinit(use_reinit);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
        1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
}

/* Initialize the value to clear the rendered images
 */
void surfelwarp::Renderer::initClearValues()
{
    m_clear_values.initialize();
}

/* The method to initialize frame/render buffer object
 */
void surfelwarp::Renderer::initFrameRenderBuffers()
{
    m_fusion_map_buffers.initialize(m_fusion_map_width, m_fusion_map_height);
    m_solver_map_buffers.initialize(m_image_width, m_image_height);
    m_visualization_draw_buffers.initialize(m_image_width, m_image_height);
}

void surfelwarp::Renderer::freeFrameRenderBuffers()
{
    m_fusion_map_buffers.release();
    m_solver_map_buffers.release();
    m_visualization_draw_buffers.release();
}

/* The access of fusion maps
 */
void surfelwarp::Renderer::MapFusionMapsToCuda(
    FusionMaps& maps, cudaStream_t stream)
{
    m_fusion_map_buffers.mapToCuda(maps.warp_vertex_map, maps.warp_normal_map,
        maps.index_map, maps.color_time_map, stream);
}

void surfelwarp::Renderer::UnmapFusionMapsFromCuda(cudaStream_t stream)
{
    m_fusion_map_buffers.unmapFromCuda(stream);
}

/* The texture access of solver maps
 */
void surfelwarp::Renderer::MapSolverMapsToCuda(
    surfelwarp::Renderer::SolverMaps& maps, cudaStream_t stream)
{
    m_solver_map_buffers.mapToCuda(maps.reference_vertex_map,
        maps.reference_normal_map, maps.warp_vertex_map, maps.warp_normal_map,
        maps.index_map, maps.normalized_rgb_map, stream);
}

void surfelwarp::Renderer::UnmapSolverMapsFromCuda(cudaStream_t stream)
{
    m_solver_map_buffers.unmapFromCuda(stream);
}
