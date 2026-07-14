set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ex1
    ex1.cpp
)

target_compile_options(ex1 PRIVATE
    $<$<CONFIG:Debug>:-g -Wall -Wextra -Wpedantic>
    $<$<CONFIG:Release>:-O3 -Wall -Wextra -Wpedantic>
)

target_link_libraries(ex1 PRIVATE
  mpoi
)

target_include_directories(ex1 PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

add_executable(ex2
    ex2.cpp
)

target_compile_options(ex2 PRIVATE
    $<$<CONFIG:Debug>:-g -Wall -Wextra -Wpedantic>
    $<$<CONFIG:Release>:-O3 -Wall -Wextra -Wpedantic>
)

target_link_libraries(ex2 PRIVATE
  mpoi
)

target_include_directories(ex2 PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

