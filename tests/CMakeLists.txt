# Unit test battery for libeisfit, built with the header-only Catch2 v2.
#
# The tests reach into the eisfit_internal namespace (bounds, diagnostics, the Problem
# types), so they link the library the same way the CLI does and include the internal
# headers from the repository root.

set(EISFIT_TEST_SOURCES
	main.cpp
	test_util.cpp
	test_bounds.cpp
	test_diagnostics.cpp
	test_fittedmodel.cpp
	test_problem.cpp
	test_fit.cpp
	test_batch.cpp
)

add_executable(${PROJECT_NAME}_tests ${EISFIT_TEST_SOURCES})
target_include_directories(${PROJECT_NAME}_tests PRIVATE
	${CMAKE_SOURCE_DIR}
	${API_HEADERS_CPP_DIR}
	${KISSTYPE_INCLUDE_DIRS}
	${EIS_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME} ${TORCH_LIBRARIES} ${KISSTYPE_LIBRARIES} ${EIS_LIBRARIES})
set_target_properties(${PROJECT_NAME}_tests PROPERTIES COMPILE_FLAGS "-Wall -O2 -g" LINK_FLAGS "-pthread")

add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)
