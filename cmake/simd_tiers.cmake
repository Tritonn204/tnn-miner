# simd_tiers.cmake -- Generate per-SIMD-tier translation units from .spec.cpp files.
#
# Usage:
#   include(cmake/simd_tiers.cmake)
#   simd_tiers_generate(
#     SPEC   src/crypto/xelis-hash/xelis-hash-v3-fmv.spec.cpp
#     OUTDIR src/crypto/xelis-hash/simd_sources
#     OUTPUT_SOURCES  my_generated_srcs   # variable name to append generated .cpp files
#   )
#
# The generated files are added as dependencies so they rebuild when the spec changes.
# The output directory is created automatically.

# Find a working Python interpreter
if(NOT SIMD_TIERS_PYTHON)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
        set(SIMD_TIERS_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python for gen_tiers.py")
    else()
        find_program(SIMD_TIERS_PYTHON NAMES py py3 python3 python)
    endif()
    if(NOT SIMD_TIERS_PYTHON)
        message(FATAL_ERROR "simd_tiers: No Python found. Set SIMD_TIERS_PYTHON.")
    endif()
    message(STATUS "simd_tiers: using '${SIMD_TIERS_PYTHON}'")
endif()

function(simd_tiers_generate)
    cmake_parse_arguments(ARG "" "SPEC;OUTDIR;OUTPUT_SOURCES" "" ${ARGN})

    if(NOT ARG_SPEC)
        message(FATAL_ERROR "simd_tiers_generate: SPEC is required")
    endif()
    if(NOT ARG_OUTDIR)
        message(FATAL_ERROR "simd_tiers_generate: OUTDIR is required")
    endif()
    if(NOT ARG_OUTPUT_SOURCES)
        message(FATAL_ERROR "simd_tiers_generate: OUTPUT_SOURCES is required")
    endif()

    # Resolve paths
    set(_spec "${PROJECT_SOURCE_DIR}/${ARG_SPEC}")
    set(_outdir "${PROJECT_SOURCE_DIR}/${ARG_OUTDIR}")
    set(_gen_script "${PROJECT_SOURCE_DIR}/scripts/gen_tiers.py")

    # Run the generator at configure time to discover output file names
    execute_process(
        COMMAND ${SIMD_TIERS_PYTHON} "${_gen_script}" --dry-run "${_spec}" "${_outdir}"
        OUTPUT_VARIABLE _gen_files
        ERROR_VARIABLE _gen_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _gen_rc
    )

    if(NOT _gen_rc EQUAL 0)
        message(STATUS "simd_tiers: dry-run failed (rc=${_gen_rc}), trying real run. stderr: ${_gen_err}")
        # Fallback: run generator for real to bootstrap
        execute_process(
            COMMAND ${SIMD_TIERS_PYTHON} "${_gen_script}" "${_spec}" "${_outdir}"
            OUTPUT_VARIABLE _gen_files
            ERROR_VARIABLE _gen_err
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _gen_rc
        )
        if(NOT _gen_rc EQUAL 0)
            message(FATAL_ERROR "simd_tiers_generate: gen_tiers.py failed for ${ARG_SPEC}\nstderr: ${_gen_err}")
        endif()
    endif()

    # Convert newline-separated list to cmake list
    string(REPLACE "\n" ";" _gen_list "${_gen_files}")

    # Filter to just .cpp files (skip .gen.h)
    set(_gen_cpps "")
    set(_gen_all "")
    foreach(_f ${_gen_list})
        string(STRIP "${_f}" _f)
        if(_f)
            list(APPEND _gen_all "${_f}")
            if(_f MATCHES "\\.gen\\.cpp$")
                list(APPEND _gen_cpps "${_f}")
            endif()
        endif()
    endforeach()

    # Custom command: re-run generator when spec changes
    add_custom_command(
        OUTPUT ${_gen_all}
        COMMAND ${SIMD_TIERS_PYTHON} "${_gen_script}" "${_spec}" "${_outdir}"
        DEPENDS "${_spec}" "${_gen_script}"
        COMMENT "Generating SIMD tier sources from ${ARG_SPEC}"
        VERBATIM
    )

    # Append generated .cpp files to the caller's variable
    set(${ARG_OUTPUT_SOURCES} ${${ARG_OUTPUT_SOURCES}} ${_gen_cpps} PARENT_SCOPE)
endfunction()
