if(USE_ASTRO_SPSA)
  if(WIN32)
    set(SPSA_OS_PREFIX "win")
  elseif(APPLE)
    set(SPSA_OS_PREFIX "macos")
  else()
    set(SPSA_OS_PREFIX "linux")
  endif()
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flto -DUSE_ASTRO_SPSA=ON")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto -DUSE_ASTRO_SPSA=ON")
  #set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -DUSE_ASTRO_SPSA=ON")
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DUSE_ASTRO_SPSA=ON")

  if(EXISTS ${PROJECT_SOURCE_DIR}/lib/astrospsa)
    set(SPSA_LIB_DIR ${PROJECT_SOURCE_DIR}/lib/astrospsa)
  else()
    include(FetchContent)
    ## Fetch the static library
    FetchContent_Declare(
        astrospsa
        GIT_REPOSITORY https://gitlab.com/Tritonn204/astro-spsa.git
        GIT_TAG        8938667bfa3253b52c622daf575fc11674d7067b
        # GIT_REPOSITORY https://gitlab.com/dirkerdero/astro-spsa-dirker.git
        # GIT_TAG        1a2acdd2ed187a9b565c33028dba4052a89f875b
    )
    FetchContent_MakeAvailable(astrospsa)
    set(SPSA_LIB_DIR ${astrospsa_SOURCE_DIR})
  endif()
  cmake_print_variables(SPSA_LIB_DIR)
  include_directories(${SPSA_LIB_DIR})

  # CMAKE_LIB_SUFFIX was set in CMakeLists.txt already
  set(_spsa_suffixes "")
  if(DEFINED CMAKE_LIB_SUFFIX AND NOT "${CMAKE_LIB_SUFFIX}" STREQUAL "")
    list(APPEND _spsa_suffixes "${CMAKE_LIB_SUFFIX}")
  endif()
  list(APPEND _spsa_suffixes "_clang_20" "_clang_19" "_clang_18" "_clang_17" "_clang_16" "")
  list(REMOVE_DUPLICATES _spsa_suffixes)

  set(SPSA_FULL_LIB_PATH "")

  foreach(_sfx IN LISTS _spsa_suffixes)
    if(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}_${CPU_ARCHTARGET}.a)
      set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}_${CPU_ARCHTARGET}.a)
      break()
    elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}.a)
      set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}.a)
      break()
    elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
      set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
      break()
    elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}.a)
      set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}.a)
      break()
    endif()
  endforeach()

  if("${SPSA_FULL_LIB_PATH}" STREQUAL "")
    message(FATAL_ERROR "SPSA lib was not found: ${SPSA_LIB_DIR}")
  endif()
endif()
