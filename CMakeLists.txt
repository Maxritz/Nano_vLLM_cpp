cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED ENV{HIP_PATH} AND EXISTS "G:/ROCM10RT-gfx1201")
  set(ENV{HIP_PATH} "G:/ROCM10RT-gfx1201")
endif()
list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(HIP_ARCHITECTURES "gfx1030;gfx1201" CACHE STRING "Target AMD GPU architectures for RDNA2/RDNA4")
set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")

project(nanovllm CXX HIP)
find_package(hip CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hip")
add_executable(nanovllm ${SOURCES})
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
target_include_directories(nanovllm PRIVATE include)
target_compile_options(nanovllm PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2 -Wno-unused-variable -Wno-unused-parameter>)
set_target_properties(nanovllm PROPERTIES HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
target_link_libraries(nanovllm PRIVATE hip::host hip::device)

add_executable(nanovllm_test src/hip_ops.hip test/test_basic.cpp)
set_source_files_properties(src/hip_ops.hip test/test_basic.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(nanovllm_test PRIVATE include)
set_target_properties(nanovllm_test PROPERTIES HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
target_link_libraries(nanovllm_test PRIVATE hip::host hip::device)

list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
add_executable(nanovllm_golden ${SOURCES} test/golden_test.cpp)
set_source_files_properties(${SOURCES} test/golden_test.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(nanovllm_golden PRIVATE include test)
target_compile_options(nanovllm_golden PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2 -Wno-unused-variable -Wno-unused-parameter>)
set_target_properties(nanovllm_golden PROPERTIES HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
target_link_libraries(nanovllm_golden PRIVATE hip::host hip::device)

# System32 holds the display driver's amdhip64_7.dll and beats PATH; delay-load it
# so src/boot_hip.cpp can point the loader at the real ROCm bin via AddDllDirectory.
function(configure_hip_runtime TARGET_NAME)
  if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE NANOVLLM_HIP_BIN="$ENV{HIP_PATH}/bin")
    target_link_options(${TARGET_NAME} PRIVATE "-Xlinker" "/DELAYLOAD:amdhip64_7.dll")
    target_link_libraries(${TARGET_NAME} PRIVATE delayimp)
  endif()
endfunction()
configure_hip_runtime(nanovllm)
configure_hip_runtime(nanovllm_test)
configure_hip_runtime(nanovllm_golden)

message(STATUS "Target AMD GPUs: ${HIP_ARCHITECTURES}")
