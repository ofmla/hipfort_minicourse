set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/c_functions.cpp PROPERTIES LANGUAGE ${kernel_lang})

add_executable (tensoradd_cfun
    ${CMAKE_CURRENT_SOURCE_DIR}/tensoradd_cfun.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_lib_c.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/c_functions.cpp)
set_target_properties(tensoradd_cfun
    PROPERTIES LINKER_LANGUAGE Fortran
)
install(TARGETS tensoradd_cfun DESTINATION bin)
