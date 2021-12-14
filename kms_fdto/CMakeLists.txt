cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(kms_glsl "kms_ftdo.vert" "kms_ftdo.frag")
set(GLSL_VALIDATOR glslangValidator)

foreach(GLSL ${kms_glsl})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${CMAKE_SOURCE_DIR}/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/"
    COMMAND ${GLSL_VALIDATOR} -x -V "${CMAKE_CURRENT_SOURCE_DIR}/${GLSL}" -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(Shaders DEPENDS ${SPIRV_BINARY_FILES})

add_compile_options(-march=armv7-a -mfloat-abi=hard -mfpu=neon -ffast-math -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-private-field -Wno-unused-variable -Wno-psabi -Werror -ggdb)

add_executable (kms_fdto "kms_fdto.cpp")

add_dependencies(kms_fdto Shaders)

target_link_libraries(kms_fdto PUBLIC vulkan pthread asound)
