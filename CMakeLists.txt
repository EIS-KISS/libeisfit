cmake_minimum_required(VERSION 3.23)

project(eisfit LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_PROJECT_VERSION_MAJOR 1)
set(CMAKE_PROJECT_VERSION_MINOR 0)
set(CMAKE_PROJECT_VERSION_PATCH 0)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(KISSTYPE REQUIRED libkisstype)
pkg_search_module(EIS REQUIRED libeisgenerator)
find_package(sciplot)

if(NOT sciplot_FOUND)
	message(WARNING "Sciplot not found, eisfit will be unable to generate plots")
else()
	add_definitions(-DENABLE_PLOTTING)
	message("Sciplot found, plotting enabled")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "..." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(SRC_FILES
	fit.cpp
	modelscript.cpp
	util.cpp
	bounds.cpp
	diagnostics.cpp
	problem.cpp
	batchproblem.cpp
)

set(API_HEADERS_CPP_DIR eisfit/)

set(API_HEADERS_CPP
	${API_HEADERS_CPP_DIR}/fit.h
)

set(COMMON_COMPILE_FLAGS "-Wall -O2 -march=native -g -Wfatal-errors")
set(COMMON_LINK_FLAGS "-pthread")

message("Linking: " "${TORCH_LIBRARIES} ${KISSTYPE_LIBRARIES} ${EIS_LIBRARIES}")

add_library(${PROJECT_NAME}_obj OBJECT ${SRC_FILES} ${API_HEADERS_CPP})
target_sources(${PROJECT_NAME}_obj INTERFACE FILE_SET header_set TYPE HEADERS BASE_DIRS ${API_HEADERS_CPP_DIR} FILES ${API_HEADERS_CPP})
target_include_directories(${PROJECT_NAME}_obj PUBLIC . ${API_HEADERS_CPP_DIR} ${KISSTYPE_INCLUDE_DIRS} ${EIS_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE 1)
set_target_properties(${PROJECT_NAME}_obj PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS ${COMMON_LINK_FLAGS})
add_library(${PROJECT_NAME} SHARED $<TARGET_OBJECTS:${PROJECT_NAME}_obj>)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${KISSTYPE_LIBRARIES} ${EIS_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
add_library(${PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${PROJECT_NAME}_obj>)
target_link_libraries(${PROJECT_NAME}_static ${TORCH_LIBRARIES} ${KISSTYPE_LIBRARIES} ${EIS_LIBRARIES})
set_target_properties(${PROJECT_NAME}_static PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})

configure_file(pkgconfig/libeisfit.pc.in pkgconfig/libeisfit.pc @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(TARGETS ${PROJECT_NAME}_static DESTINATION lib)
install(TARGETS ${PROJECT_NAME}_obj FILE_SET header_set DESTINATION include/${PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/libeisfit.pc DESTINATION lib/pkgconfig)

add_executable(${PROJECT_NAME}_cli main.cpp log.cpp ploting.cpp)
set_target_properties(${PROJECT_NAME}_cli PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
add_dependencies(${PROJECT_NAME}_cli ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_cli ${PROJECT_NAME})
if(sciplot_FOUND)
	target_link_libraries(${PROJECT_NAME}_cli sciplot::sciplot)
endif()
target_include_directories(${PROJECT_NAME}_cli PUBLIC . ${API_HEADERS_CPP_DIR})
set_target_properties(${PROJECT_NAME}_cli PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS ${COMMON_LINK_FLAGS})
install(TARGETS ${PROJECT_NAME}_cli DESTINATION bin)

option(EISFIT_BUILD_TESTS "Build the libeisfit unit test battery" ON)
if(EISFIT_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
