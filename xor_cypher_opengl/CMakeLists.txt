cmake_minimum_required(VERSION 3.5)

project(xor_cypher_gpu)

set(CMAKE_BUILD_TYPE Debug)
include_directories(include usr/include)
file(GLOB HEADERS include/*.h)
file(GLOB SOURCES src/*.cpp)
find_package (Threads)
set(PROJECT_LINK_LIBS libGLEW.so libGL.so libGLU.so libglfw.so.3 libassimp.so -lpthread)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_LINK_LIBS})

