#include "Shader.h"


bool Shader::loadShaders(const char * vsFilePath, const char * fsFilePath, const char* gsFilePath)
{
	std::string vsString = fileToString(vsFilePath);
	std::string fsString = fileToString(fsFilePath);
	const GLchar* vsCode = vsString.c_str();
	const GLchar* fsCode = fsString.c_str();

	GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vShader, 1, &vsCode, nullptr);
	glCompileShader(vShader);
    checkCompileErrors(vShader);

	GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fShader, 1, &fsCode, nullptr);
	glCompileShader(fShader);
    checkCompileErrors(fShader);

	GLuint gShader = 0;
	if (gsFilePath != nullptr)
	{
        std::string gsString = fileToString(gsFilePath);
        const GLchar* gsCode = gsString.c_str();

        gShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(gShader, 1, &gsCode, nullptr);
        glCompileShader(gShader);
        checkCompileErrors(gShader);
	}


	ID = glCreateProgram();
	glAttachShader(ID, vShader);
	glAttachShader(ID, fShader);
	if(gsFilePath != nullptr) glAttachShader(ID, gShader);
	glLinkProgram(ID);
    checkLinkErrors();

    getAttribInfo();
    getUniformInfo();

	glDeleteShader(vShader);
	glDeleteShader(fShader);
    if(gsFilePath) glDeleteShader(gShader);
	uniformLocation.clear();

    return true;
}

bool Shader::loadComputeShader(const char *csFilePath)
{
    std::string csString = fileToString(csFilePath);
    const GLchar* csCode = csString.c_str();

    GLuint cShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cShader, 1, &csCode, nullptr);
    glCompileShader(cShader);
    checkCompileErrors(cShader);

    ID = glCreateProgram();
    glAttachShader(ID, cShader);
    glLinkProgram(ID);
    checkLinkErrors();

    getAttribInfo();
    getUniformInfo();

    glDeleteShader(cShader);
    uniformLocation.clear();

    return true;
}

void Shader::useProgram()
{
	if (ID > 0)
		glUseProgram(ID);
}

void Shader::validateProgram() {
    GLint result = 0;
    GLchar eLog[1024] = { 0 };
    glValidateProgram(ID);
    glGetProgramiv(ID, GL_VALIDATE_STATUS, &result);
    if (!result) {
        glGetProgramInfoLog(ID, sizeof(eLog), nullptr, eLog);
        printf("Error validating program: %s\n", eLog);
        exit(EXIT_FAILURE);
    }
}

std::string Shader::fileToString(const std::string & filePath)
{
	std::stringstream ss;
	std::ifstream file;
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		file.open(filePath, std::ios::in);
		if (file.is_open())
			ss << file.rdbuf();
		file.close();
	}
	catch (std::ifstream::failure ex)
	{
		std::cerr << "Error reading shader file!\n";
	}
	return ss.str();
}

void Shader::checkCompileErrors(GLuint shader) const
{
	int status = 0;
	GLint length = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string errorLog(static_cast<unsigned long>(length), ' ');
        glGetShaderInfoLog(ID, length, &length, &errorLog[0]);
        std::cerr << "Error compiling shader.\n" << errorLog << std::endl;
    }
}

void Shader::checkLinkErrors() const
{
    int status = 0;
    GLint length = 0;
    glGetProgramiv(ID, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramiv(ID, GL_INFO_LOG_LENGTH, &length);
        std::string errorLog(static_cast<unsigned long>(length), ' ');
        glGetProgramInfoLog(ID, length, &length, &errorLog[0]);
        std::cerr << "Error linking program.\n" << errorLog << std::endl;
    }
}

GLint Shader::getUniformLocation(const GLchar * name)
{
	auto it = uniformLocation.find(name);
	if (it == uniformLocation.end())
	{
		uniformLocation[name] = glGetUniformLocation(ID, name);
	}

	return uniformLocation[name];
}

