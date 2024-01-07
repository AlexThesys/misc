#include <memory>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <limits>
#include <functional>
#include <thread>
#include <vector>
#include <iostream>
#include <cstring>
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "Shader.h"

using namespace std;

static constexpr int MAX_CYCLES = 256;

static void printWorkgroupsInfo();
template <typename T>
static void setupTextureImage(GLuint& texBuf, T* buffer, const size_t size, const int binding, const GLenum);
template <typename T>
static void getImageData(const GLuint texBuf, T* buffer, const size_t size);
template <typename T>
void setUniformBuffer(const Shader& shader, const T* keys, const int num_keys);

class Random
    {
    public:
        Random(int mx = numeric_limits<int>::max(), int mn = 0) : _max(mx), _min(mn), A(13), B(7), M(11), x(51){}
       constexpr int random(int mx = numeric_limits<int>::max(), int mn = 0) noexcept
        {
            _max = mx;
            _min = mn;

            x = (A * x + B) % M;

           return static_cast<int>(_min + ((float)x / (float)M) * (_max - _min));
        }

        constexpr int operator()(int mx = numeric_limits<int>::max(), int mn = 0) noexcept
        {
            _max = mx;
            _min = mn;

            x = (A * x + B) % M;

           return static_cast<int>(_min + ((float)x / (float)M) * (_max - _min));
        }

       void seed(int _x) noexcept {x = _x;}
    private:
       int x;
       int _max, _min;
      const int A, B, M;
    };

class FCypherThreaded
{
public:
    FCypherThreaded(void* arr, const int cycles,
                    const int n, const int block_size);
    enum class ACTION {CYPHER, DECYPHER};
    void process(FCypherThreaded::ACTION);
    GLuint getTexBufID() const noexcept { return m_texBuf; }
private:
    Shader shader;
    Random rnd;
    struct {int numBlocks, blockSize;} m_workgroup;
    int m_size, m_cycles;
    GLuint m_texBuf = 0u;
};

FCypherThreaded::FCypherThreaded(void* arr, const int cycles,
                                 const int n, const int block_size = 256) :
                                       m_size(n),
                                       m_cycles(cycles)
{
    assert((m_size & 0x1) == 0);
    assert(m_cycles <= MAX_CYCLES);

    shader.loadComputeShader("../Shaders/shader.comp");
    shader.useProgram();
    shader.setUniform("num_cycles", m_cycles);
    glUseProgram(0);

    setupTextureImage(m_texBuf, static_cast<int*>(arr), static_cast<size_t>(m_size), 0, GL_READ_WRITE);

    rnd.seed(static_cast<int>(m_cycles<<1));
    std::unique_ptr<int[]> keys = unique_ptr<int[]>(new int[static_cast<unsigned>(m_cycles)]);
    for (int i = 0 ; i < m_cycles; ++i)
        keys[static_cast<size_t>(i)] = rnd.random(m_cycles);

    setUniformBuffer(shader, keys.get(), m_cycles);
    m_workgroup.blockSize = block_size;
    assert((m_size & (m_workgroup.blockSize - 1)) == 0);
    m_workgroup.numBlocks = m_size / (m_workgroup.blockSize * 2);
}

void FCypherThreaded::process(FCypherThreaded::ACTION type)
{
    shader.useProgram();
    shader.setUniform("action", static_cast<int>(type));
    glDispatchCompute(m_workgroup.numBlocks, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glUseProgram(0);
}

static void setupGLEW()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* mainWindow = glfwCreateWindow(64, 64, "Test", nullptr, nullptr);
    if (mainWindow == nullptr) {
        puts("Failed to create GLFW widnow! Exiting...");
        glfwTerminate();
    }
    glfwMakeContextCurrent(mainWindow);

    glewExperimental = GL_TRUE;
    GLenum glewError = glewInit();
    if (glewError != GLEW_OK)
        printf("glew initialization error: %d\n", glewError);
}


int main()
{
//    printWorkgroupsInfo();
    setupGLEW();
    Random rnd;
    constexpr size_t size = 2048 << 2;
    vector<int> buffer(size);
    for (size_t i = 0; i < size; ++i)
        buffer[i] = rnd.random(4096);
//    for (auto i = 0; i < 8; ++i)
//        cout << buffer[i] << "\n";
//    cout << "***\n";

    FCypherThreaded fct(buffer.data(), 32, size);

    fct.process(FCypherThreaded::ACTION::CYPHER);
    getImageData(fct.getTexBufID(), buffer.data(), 32);
//    for (auto i = 0; i < 8; ++i)
//        cout << buffer[i] << "\n";
//    cout << "***\n";

    fct.process(FCypherThreaded::ACTION::DECYPHER);
    getImageData(fct.getTexBufID(), buffer.data(), 32);
//    for (auto i = 0; i < 8; ++i)
//        cout << buffer[i] << "\n";
//    cout << "***\n";

    return 0;
}

template <typename T>
static void setupTextureImage(GLuint& texBuf, T* buffer, const size_t size,
                              const int binding, const GLenum mode)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glGenBuffers(1, &texBuf);
    glBindBuffer(GL_TEXTURE_BUFFER, texBuf);
    glBufferData(GL_TEXTURE_BUFFER, sizeof (T) * size, nullptr, GL_STREAM_COPY);
    T* device_mem = (T*) glMapBufferRange(GL_TEXTURE_BUFFER, 0, size*sizeof (T),
                                          GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    memcpy(device_mem, buffer, sizeof (T) * size);
    glUnmapBuffer(GL_TEXTURE_BUFFER);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, texBuf);
    glBindImageTexture(binding, tex, 0, GL_FALSE, 0, mode, GL_R32I);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

}

template <typename T>
void setUniformBuffer(const Shader& shader, const T* keys, const int num_keys)
{
    assert(sizeof (T) == 4);
    GLuint ubo;
    constexpr GLint bindPoint = 0;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, MAX_CYCLES * (sizeof (T)), nullptr, GL_STREAM_COPY);
    T* device_mem = (T*) glMapBufferRange(GL_UNIFORM_BUFFER, 0, num_keys*sizeof (T),
                                          GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    memcpy(device_mem, keys, sizeof (T) * num_keys);
    glUnmapBuffer(GL_UNIFORM_BUFFER);
    GLuint index = glGetUniformBlockIndex(shader.getProgramID(), "keys_block");
    glUniformBlockBinding(shader.getProgramID(), index, bindPoint);
    glBindBufferRange(GL_UNIFORM_BUFFER, bindPoint, ubo, 0, MAX_CYCLES * sizeof (T));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

template <typename T>
static void getImageData(const GLuint texBuf, T* buffer, const size_t size)
{
    glBindBuffer(GL_TEXTURE_BUFFER, texBuf);
    T* device_data = (T*)glMapBuffer(GL_TEXTURE_BUFFER, GL_READ_ONLY);
    memcpy(buffer, device_data, size * sizeof (T));
    glUnmapBuffer(GL_TEXTURE_BUFFER);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

static void printWorkgroupsInfo()
{
    // number of workgroups
    int work_grp_cnt[3];

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);

    printf("max global (total) work group size x:%i y:%i z:%i\n",
           work_grp_cnt[0], work_grp_cnt[1], work_grp_cnt[2]);

    // size of a workgroup
    int work_grp_size[3];

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);

    printf("max local (in one shader) work group sizes x:%i y:%i z:%i\n",
           work_grp_size[0], work_grp_size[1], work_grp_size[2]);
}
