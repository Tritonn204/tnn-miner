if (WITH_ASTRIXHASH)
    add_definitions(/DTNN_ASTRIXHASH)

    message(STATUS "Building with AstrixHash support")

    file(GLOB_RECURSE astrixHashHeaders
      src/crypto/astrix-hash/*.h
    )

    file(GLOB_RECURSE astrixHashSources
      src/crypto/astrix-hash//*.cpp
      src/crypto/astrix-hash//*.c
      src/net/astrix/*.cpp
      # src/net/proto/astrix/*.cc
    )
    
    list(APPEND astrixHashSources
      src/coins/mine_astrix.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${astrixHashHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${astrixHashSources}
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
    remove_definitions(/DTNN_ALGO_ASTRIXHASH)
endif()
unset(WITH_ASTRIXHASH CACHE)
