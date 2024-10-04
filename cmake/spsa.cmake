if(USE_ASTRO_SPSA)
  set(SPSA_OS_PREFIX "linux")
  if(WIN32)
    set(SPSA_OS_PREFIX "win")
  endif()
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flto -DUSE_ASTRO_SPSA=ON")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto -DUSE_ASTRO_SPSA=ON")

  if(EXISTS ${PROJECT_SOURCE_DIR}/lib/astrospsa/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
    set(SPSA_LIB_DIR ${PROJECT_SOURCE_DIR}/lib/astrospsa)
  else()
    include(FetchContent)
    ## Fetch the static library
    FetchContent_Declare(
        astrospsa
        GIT_REPOSITORY https://gitlab.com/Tritonn204/astro-spsa.git
        GIT_TAG        643809adc68d5fcf716ec45f5452f6d5c50abac1
    )
    FetchContent_MakeAvailable(astrospsa)
    set(SPSA_LIB_DIR ${astrospsa_SOURCE_DIR})
  endif()
  cmake_print_variables(SPSA_LIB_DIR)
  include_directories(${SPSA_LIB_DIR})
endif()