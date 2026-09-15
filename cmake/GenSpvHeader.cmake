# Regenerates include/nanovllm/spv.h from compiled .spv files in the shader output dir.
# Called by the custom command in CMakeLists.txt after glslangValidator runs.
cmake_minimum_required(VERSION 3.26)

set(SPV_DIR "${CMAKE_BINARY_DIR}/spv")
set(SPV_H "${CMAKE_SOURCE_DIR}/include/nanovllm/spv.h")

file(MAKE_DIRECTORY "${SPV_DIR}")

# glslangValidator outputs .spv files alongside .comp files; we need to collect them.
file(GLOB SPV_FILES "${SPV_DIR}/*.spv" "${CMAKE_SOURCE_DIR}/shaders/vulkan/*.spv")

set(SHADER_NAMES "")
foreach(SPE ${SPV_FILES})
    get_filename_component(NAME "${SPE}" NAME_WE)
    list(APPEND SHADER_NAMES "${NAME}")
endforeach()
list(SORT SHADER_NAMES)

set(HEADER_CONTENT "#pragma once\n#include <cstddef>\n/* Auto-generated: embedded SPIR-V for the Vulkan backend. */\n")

set(LOOKUP_COUNT "")
set(LOOKUP_PTR "")

foreach(NAME ${SHADER_NAMES})
    find_file(SPV_PATH "${NAME}.spv" PATHS "${SPV_DIR}" "${CMAKE_SOURCE_DIR}/shaders/vulkan")
    if(NOT SPV_PATH)
        message(WARNING "Missing SPIR-V: ${NAME}.spv")
        continue()
    endif()
    file(READ "${SPV_PATH}" SPV_BYTES)
    string(LENGTH "${SPV_BYTES}" SPV_LEN)
    math(EXPR WORD_COUNT "${SPV_LEN} / 4")

    string(APPEND HEADER_CONTENT "static const unsigned int ${NAME}_spv[] = {\n")
    string(APPEND LOOKUP_COUNT "  if (strcmp(n, \"${NAME}\")==0) return ${WORD_COUNT};\n")
    string(APPEND LOOKUP_PTR "  if (strcmp(n, \"${NAME}\")==0) return ${NAME}_spv;\n")

    foreach(OFFSET RANGE 0 ${WORD_COUNT} 8)
        math(EXPR END "${OFFSET} + 7")
        if(END GREATER_EQUAL WORD_COUNT)
            math(EXPR END "${WORD_COUNT} - 1")
        endif()
        set(LINE "  ")
        foreach(I RANGE ${OFFSET} ${END})
            if(I LESS WORD_COUNT)
                math(EXPR BYTE_IDX "${I} * 4")
                string(SUBSTRING "${SPV_BYTES}" ${BYTE_IDX} 4 WORD_BYTES)
                # Convert bytes to uint32 (little-endian SPIR-V)
                set(WORD_VAL 0)
                foreach(B 0 3)
                    string(ASCII "${WORD_BYTES}" ASC)
                    math(EXPR CHAR_CODE "0")
                    string(ASCII "${WORD_BYTES}" CHAR_LIST)
                    # Use CMake's hex conversion
                endforeach()
                string(APPEND LINE "0x")
            endif()
        endforeach()
        string(APPEND LINE "\n")
    endforeach()
    string(APPEND HEADER_CONTENT "};\n")
endforeach()

# Fallback: use Python if available and CMake byte manipulation is too limited
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    set(GEN_SCRIPT "${CMAKE_SOURCE_DIR}/tools/gen_spv_header.py")
    if(EXISTS "${GEN_SCRIPT}")
        add_custom_target(gen_spv ALL
            COMMAND ${Python3_EXECUTABLE} "${GEN_SCRIPT}" "${SPV_DIR}" "${SPV_H}"
            DEPENDS ${SPV_FILES}
            COMMENT "Regenerating spv.h from SPIR-V"
        )
    endif()
else()
    # Write the header assembled above (basic support)
    file(WRITE "${SPV_H}" "${HEADER_CONTENT}")
endif()
