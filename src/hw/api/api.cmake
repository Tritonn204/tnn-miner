if (WITH_HTTP)
    add_definitions(/DTNN_FEATURE_DMI)

    list(APPEND HEADERS
        src/hw/api/HwApi.h
        )

    list(APPEND SOURCES
        src/hw/api/HwApi.cpp
        )
endif()
