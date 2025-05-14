if (WITH_ASTROBWTV3)
    add_definitions(/DTNN_ASTROBWTV3)

    message(STATUS "Building with AstroBWTv3 support")

    file(GLOB_RECURSE astroHeaders
      src/crypto/astrobwtv3/*.h
      src/crypto/spectrex/*.h
    )

    file(GLOB_RECURSE astroSources
      src/crypto/astrobwtv3/*.cpp
      src/crypto/astrobwtv3/*.c
      src/crypto/spectrex/*.cpp
      src/crypto/spectrex/*.c
      src/net/dero/*.cpp
      src/net/spectre/*.cpp
    )

    list(FILTER astroSources EXCLUDE REGEX ".*bench\\.cpp$")
    list(FILTER astroSources EXCLUDE REGEX ".*/build/.*")
    
    list(APPEND astroSources
      src/coins/mine_dero.cpp
      src/coins/mine_spectre.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${astroHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${astroSources}
    )

    # if (WITH_MSR AND NOT TNN_ARM AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND (TNN_OS_WIN OR TNN_OS_LINUX))
    #     add_definitions(/DTNN_FEATURE_MSR)
    #     add_definitions(/DTNN_FIX_RYZEN)
    #     message("-- WITH_MSR=ON")

    #     if (TNN_OS_WIN)
    #         list(APPEND SOURCES_CRYPTO
    #             src/crypto/rx/RxFix_win.cpp
    #             src/hw/msr/Msr_win.cpp
    #             )
    #     elseif (TNN_OS_LINUX)
    #         list(APPEND SOURCES_CRYPTO
    #             src/crypto/rx/RxFix_linux.cpp
    #             src/hw/msr/Msr_linux.cpp
    #             )
    #     endif()

    #     list(APPEND HEADERS_CRYPTO
    #         src/crypto/rx/RxFix.h
    #         src/crypto/rx/RxMsr.h
    #         src/hw/msr/Msr.h
    #         src/hw/msr/MsrItem.h
    #         )

    #     list(APPEND SOURCES_CRYPTO
    #         src/crypto/rx/RxMsr.cpp
    #         src/hw/msr/Msr.cpp
    #         src/hw/msr/MsrItem.cpp
    #         )
    # else()
    #     remove_definitions(/DTNN_FEATURE_MSR)
    #     remove_definitions(/DTNN_FIX_RYZEN)
    #     message("-- WITH_MSR=OFF")
    # endif()

else()
    remove_definitions(/DTNN_ASTROBWTV3)
endif()
