cmake_minimum_required(VERSION 3.23)
project(ctorch)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_DEBUG
   "-g -O0 -Wall -Wextra -Wpedantic -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wextra -Wpedantic")
set(CMAKE_C_COMPILER clang)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


file(GLOB_RECURSE SOURCES "src/*.c")

add_library(ctorch_src STATIC ${SOURCES})
target_include_directories(ctorch_src PUBLIC include)
target_link_libraries(ctorch_src PUBLIC m)

add_executable(ctorch main.c)
target_link_libraries(ctorch ctorch_src)

