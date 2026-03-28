if (APPLE)
    message(STATUS "ANE backend: enabled (Apple Neural Engine)")

    ggml_add_backend_library(ggml-ane
        ggml-ane.mm
        ane_bridge.m
    )

    target_include_directories(ggml-ane PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(ggml-ane PRIVATE
        ggml-base
        "-framework Foundation"
        "-framework IOSurface"
        "-ldl"
    )

    target_compile_options(ggml-ane PRIVATE -fobjc-arc)
endif()
