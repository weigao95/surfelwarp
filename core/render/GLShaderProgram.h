//
// Created by wei on 3/26/18.
//

#pragma once

#include "common/global_configs.h"
#include "core/render/glad/glad.h"
#include "common/common_types.h"

namespace surfelwarp {
	
	class GLShaderProgram {
	private:
		GLuint m_vertex_shader;
		GLuint m_fragment_shader;
		GLuint m_geometry_shader;
		GLuint m_program_id;
	public:
		//Constructor and destructor
		explicit GLShaderProgram();
		~GLShaderProgram();
		
		
		/* The method to compile shader program from source
		 */
		void Compile(const std::string& vertex_path, const std::string& fragment_path);
		
		//The input is shader source, not
		//the path to shader file
		void Compile(
			const char* vertex_shader,
			const char* fragment_shader,
			const char* geometry_shader = nullptr
		);
	
	private:
		static void checkShaderCompilerError(GLuint shader, const std::string& type);
	
		/* The methods to use and setup uniform values
		 */
	public:
		void Bind() const { glUseProgram(m_program_id); }
		void Unbind() const { glUseProgram(0); }
		GLuint ProgramID() const { return m_program_id; }
		
		//Set the uniform value used in shader
		void SetUniformMatrix(const char* name, const Matrix4f& value);
		void SetUniformVector(const char* name, const Vector4f& vec);
		void SetUniformVector(const char* name, const float4& vec);
		void SetUniformVector(const char* name, const float3& vec);
		void SetUniformVector(const char* name, const float2& vec);
		void SetUniformFloat(const char* name, const float value);
	};

}
