//
// Created by wei on 3/26/18.
//

#include "common/logging.h"
#include "core/render/GLShaderProgram.h"
#include <fstream>
#include <sstream>

surfelwarp::GLShaderProgram::GLShaderProgram() :
	m_vertex_shader(0),
	m_fragment_shader(0),
	m_geometry_shader(0),
	m_program_id(0)
{ }

surfelwarp::GLShaderProgram::~GLShaderProgram() {
	glDeleteProgram(m_program_id);
}


/* The methods to compile the shader
 */
void surfelwarp::GLShaderProgram::Compile(const std::string &vertex_path, const std::string &fragment_path) {
	std::string vertex_code;
	std::string fragment_code;
	std::ifstream vertexShaderFile;
	std::ifstream fragmentShaderFile;
	vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		vertexShaderFile.open(vertex_path);
		fragmentShaderFile.open(fragment_path);
		std::stringstream vertex_stream, fragment_stream;
		vertex_stream << vertexShaderFile.rdbuf();
		fragment_stream << fragmentShaderFile.rdbuf();
		vertexShaderFile.close();
		fragmentShaderFile.close();
		vertex_code = vertex_stream.str();
		fragment_code = fragment_stream.str();
	}
	catch (std::ifstream::failure e) {
		LOG(FATAL) << "Error: the shader file is not correctly loaded";
	}
	vertex_code.push_back('\0');
	fragment_code.push_back('\0');
	Compile(vertex_code.c_str(), fragment_code.c_str());
}

void
surfelwarp::GLShaderProgram::Compile(const char *vertex_shader, const char *fragment_shader, const char *geometry_shader) {
	//Create and compile a vertex shader
	m_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(m_vertex_shader, 1, &vertex_shader, NULL);
	glCompileShader(m_vertex_shader);
	checkShaderCompilerError(m_vertex_shader, "VertexShader");
	
	//For a fragment shader
	m_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(m_fragment_shader, 1, &fragment_shader, NULL);
	glCompileShader(m_fragment_shader);
	checkShaderCompilerError(m_fragment_shader, "FragmentShader");
	
	//For geometry shader
	if (geometry_shader != nullptr) {
		m_geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(m_geometry_shader, 1, &geometry_shader, NULL);
		glCompileShader(m_geometry_shader);
		checkShaderCompilerError(m_geometry_shader, "GeometryShader");
	}
	
	//Link to shader program
	m_program_id = glCreateProgram();
	glAttachShader(m_program_id, m_vertex_shader);
	glAttachShader(m_program_id, m_fragment_shader);
	if (m_geometry_shader != 0) glAttachShader(m_program_id, m_geometry_shader);
	glLinkProgram(m_program_id);
	checkShaderCompilerError(m_program_id, "ShaderProgram");
	
	//As they are already linked to the program
	glDeleteShader(m_vertex_shader);
	glDeleteShader(m_fragment_shader);
	if (m_geometry_shader != 0)
		glDeleteShader(m_geometry_shader);
}


void surfelwarp::GLShaderProgram::checkShaderCompilerError(GLuint shader, const std::string &type) {
	int success;
	char infoLog[1024];
	if (type != "ShaderProgram")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			LOG(FATAL) << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			LOG(FATAL) << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

/* A series of methods to setup uniform value
 */
void surfelwarp::GLShaderProgram::SetUniformMatrix(const char *name, const surfelwarp::Matrix4f &value) {
	auto uniform_loc = glGetUniformLocation(ProgramID(), name);
	glUniformMatrix4fv(uniform_loc, 1, GL_FALSE, value.data());
}

void surfelwarp::GLShaderProgram::SetUniformVector(const char *name, const surfelwarp::Vector4f &vec) {
	auto uniform_loc = glGetUniformLocation(ProgramID(), name);
	glUniform4f(uniform_loc, vec[0], vec[1], vec[2], vec[3]);
}

void surfelwarp::GLShaderProgram::SetUniformVector(const char *name, const float4 &vec) {
	auto uniform_loc = glGetUniformLocation(ProgramID(), name);
	glUniform4f(uniform_loc, vec.x, vec.y, vec.z, vec.w);
}

void surfelwarp::GLShaderProgram::SetUniformVector(const char *name, const float3 &vec) {
	auto uniform_loc = glGetUniformLocation(ProgramID(), name);
	glUniform3f(uniform_loc, vec.x, vec.y, vec.z);
}

void surfelwarp::GLShaderProgram::SetUniformVector(const char *name, const float2 &vec) {
	auto uniform_loc = glGetUniformLocation(ProgramID(), name);
	glUniform2f(uniform_loc, vec.x, vec.y);
}

void surfelwarp::GLShaderProgram::SetUniformFloat(const char *name, const float value) {
	auto uniform_loc = glGetUniformLocation(ProgramID(), name);
	glUniform1f(uniform_loc, value);
}


