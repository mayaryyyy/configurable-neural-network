cmake_minimum_required(VERSION 3.15)
project(ConfigurableNeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(configurable_nn
    src/main.cpp
    src/Activation.cpp
    src/Neuron.cpp
    src/Layer.cpp
    src/Network.cpp
    src/ConfigParser.cpp
    src/Dataset.cpp
    src/BenchmarkRunner.cpp
)

target_include_directories(configurable_nn PRIVATE include)

if(MSVC)
    target_compile_options(configurable_nn PRIVATE /W4)
else()
    target_compile_options(configurable_nn PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Copy the config/ directory next to the built executable so the program
# can find its configuration files using a relative path at runtime,
# regardless of which build system generator is used.
add_custom_command(TARGET configurable_nn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config
        $<TARGET_FILE_DIR:configurable_nn>/config
)
