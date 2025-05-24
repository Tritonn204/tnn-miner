if (WITH_YESPOWER)
    add_definitions(/DTNN_YESPOWER)

    message(STATUS "Building with YesPower support")

    file(GLOB_RECURSE ypHeaders
      src/crypto/yespower/*.h
    )

    file(GLOB_RECURSE ypSources
      src/crypto/yespower/*.cpp
      src/crypto/yespower/*.c
      src/net/btc/*.cpp
    )
    
    list(APPEND ypSources
      src/coins/mine_yespower.cpp
    )

    list(APPEND HEADERS_CRYPTO
      ${ypHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${ypSources}
    )

else()
    remove_definitions(/DTNN_YESPOWER)
endif()
