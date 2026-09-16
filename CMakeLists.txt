cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED ENV{HIP_PATH} AND EXISTS "G:/ROCM10RT-gfx1201")
  set(ENV{HIP_PATH} "G:/ROCM10RT-gfx1201")
endif()
list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(HIP_ARCHITECTURES "gfx1031;gfx1201" CACHE STRING "Target AMD GPU architectures for RDNA2/RDNA4")

# HIP is optional: build HIP targets only if ROCm is found. Vulkan targets
# build on any platform with the Vulkan SDK installed.
find_package(hip CONFIG QUIET)

project(nanovllm CXX)

if(hip_FOUND)
  set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
  enable_language(HIP)
  message(STATUS "HIP found: ${hip_VERSION} — building AMD ROCm targets")

  file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hip")
  # Vulkan backend sources are plain C++ (host compiler), not HIP. Exclude them from
  # the HIP-globbed SOURCES list so they aren't compiled with hipcc.
  list(REMOVE_ITEM SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_backend.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_model.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/main_vulkan.cpp")
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

  # HIP single-sequence top-5 dump (adjudication/debug tool, no Python needed).
  add_executable(hip_seq_top5 ${SOURCES} tools/hip_seq_top5.cpp)
  set_source_files_properties(${SOURCES} tools/hip_seq_top5.cpp PROPERTIES LANGUAGE HIP)
  target_include_directories(hip_seq_top5 PRIVATE include test)
  target_compile_options(hip_seq_top5 PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2 -Wno-unused-variable -Wno-unused-parameter>)
  set_target_properties(hip_seq_top5 PROPERTIES HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")
  target_link_libraries(hip_seq_top5 PRIVATE hip::host hip::device)
  configure_hip_runtime(hip_seq_top5)
else()
  message(STATUS "HIP not found — skipping AMD ROCm targets (Vulkan target still available)")
endif()

  # Host fp32 prefill reference (exact-math adjudicator, no GPU): same loader path,
  # plain C++ like the Vulkan targets (HIP headers only, no HIP link).
  add_executable(ref_top5 tools/ref_top5.cpp src/gguf.cpp src/safetensors.cpp src/tokenizer.cpp)
  target_include_directories(ref_top5 PRIVATE include)
  if(DEFINED ENV{HIP_PATH})
    target_include_directories(ref_top5 PRIVATE "$ENV{HIP_PATH}/include")
  endif()
  target_compile_options(ref_top5 PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/O2 /wd4100 /wd4101>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wno-unused-variable -Wno-unused-parameter -Wno-deprecated-declarations>)

message(STATUS "Target AMD GPUs: ${HIP_ARCHITECTURES}")

# ── Vulkan backend (system Vulkan SDK; no external VAiT dependency) ────────────
find_package(Vulkan REQUIRED)
add_executable(nanovllm_vulkan src/vulkan_backend.cpp src/vulkan_model.cpp src/main_vulkan.cpp
    src/gguf.cpp src/safetensors.cpp src/tokenizer.cpp)
target_include_directories(nanovllm_vulkan PRIVATE include "${Vulkan_INCLUDE_DIR}")
target_compile_options(nanovllm_vulkan PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /wd4100 /wd4101>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wno-unused-variable -Wno-unused-parameter -Wno-deprecated-declarations>)
  target_link_libraries(nanovllm_vulkan PRIVATE Vulkan::Vulkan)

  # Shader recompilation: regenerate spv.h when .comp files change.
  find_program(GLSLANG_VALIDATOR glslangValidator PATHS "C:/VulkanSDK")
  set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders/vulkan")
  set(SPV_H "${CMAKE_SOURCE_DIR}/include/nanovllm/spv.h")
  if(GLSLANG_VALIDATOR)
    file(GLOB SHADERS "${SHADER_DIR}/*.comp")
    add_custom_command(
        OUTPUT "${SPV_H}"
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 ${SHADERS}
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_SOURCE_DIR}/cmake/GenSpvHeader.cmake"
        DEPENDS ${SHADERS} "${CMAKE_SOURCE_DIR}/src/vulkan_model.cpp"
        COMMENT "Compiling GLSL shaders to SPIR-V and regenerating spv.h"
    )
  else()
    message(STATUS "glslangValidator not found — using prebuilt spv.h")
  endif()

# AGENT-1 / REAS-1 / GEN-1: header self-tests (each header ships its own
# main() behind a *_TEST macro; compile each in isolation to run its asserts).
add_executable(agent_test test/agent_test.cpp)
target_include_directories(agent_test PRIVATE include)
add_executable(reason_test test/reason_test.cpp)
target_include_directories(reason_test PRIVATE include)
add_executable(gen_test test/gen_test.cpp)
target_include_directories(gen_test PRIVATE include)
add_executable(lora_test test/lora_test.cpp src/gguf.cpp)
target_include_directories(lora_test PRIVATE include)
target_compile_options(lora_test PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /wd4100 /wd4101>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wno-unused-variable -Wno-unused-parameter -Wno-deprecated-declarations>)
add_executable(drafter_test test/drafter_test.cpp)
target_include_directories(drafter_test PRIVATE include)
add_executable(cache_test test/cache_test.cpp)
target_include_directories(cache_test PRIVATE include)
add_executable(vk_golden_test test/vk_golden_test.cpp src/vulkan_backend.cpp src/vulkan_model.cpp
    src/gguf.cpp src/safetensors.cpp src/tokenizer.cpp)
target_include_directories(vk_golden_test PRIVATE include test "${Vulkan_INCLUDE_DIR}")
target_compile_options(vk_golden_test PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /wd4100 /wd4101>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wno-unused-variable -Wno-unused-parameter -Wno-deprecated-declarations>)
target_link_libraries(vk_golden_test PRIVATE Vulkan::Vulkan)
