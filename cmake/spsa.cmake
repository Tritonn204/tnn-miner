if(USE_ASTRO_SPSA)
  set(SPSA_OS_PREFIX "linux")
  if(WIN32)
    set(SPSA_OS_PREFIX "win")
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
        GIT_TAG        25e68cf63ce7b188a80f955dd0637d676d8c778e
        # GIT_REPOSITORY https://gitlab.com/dirkerdero/astro-spsa-dirker.git
        # GIT_TAG        1a2acdd2ed187a9b565c33028dba4052a89f875b
    )
    FetchContent_MakeAvailable(astrospsa)
    set(SPSA_LIB_DIR ${astrospsa_SOURCE_DIR})
  endif()
  cmake_print_variables(SPSA_LIB_DIR)
  include_directories(${SPSA_LIB_DIR})

  # CMAKE_LIB_SUFFIX was set in CMakeLists.txt already
  if(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${CMAKE_LIB_SUFFIX}_${CPU_ARCHTARGET}.a)
    set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${CMAKE_LIB_SUFFIX}_${CPU_ARCHTARGET}.a)
  elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${CMAKE_LIB_SUFFIX}.a)
    set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${CMAKE_LIB_SUFFIX}.a)
  elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
    set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
  elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}.a)
    set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}.a)
  else()
    message(FATAL_ERROR "SPSA lib was not found: ${SPSA_LIB_DIR}")
  endif()
endif()
