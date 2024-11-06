#include_directories(${ROCM_PATH}include)

# Set all kernel sources 
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
)
# Set the language of kernel sources
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE ${kernel_lang})
add_library(kernels_laplacian_example OBJECT "${KERNEL_FILES}")
target_include_directories(kernels_laplacian_example PRIVATE /opt/rocm/include)

# tensoradd_hip_fptr - Tensor addition with HIP and Fortran pointers
add_executable (laplacian_hip_fptr
    ${CMAKE_CURRENT_SOURCE_DIR}/main_hipfort.f90
    $<TARGET_OBJECTS:kernels_laplacian_example>
)
# Link in other libraries
target_link_libraries(laplacian_hip_fptr ${kernel_libs})
# This is needed so that the linker used is the Fortran compiler
set_target_properties(laplacian_hip_fptr
    PROPERTIES LINKER_LANGUAGE Fortran
)
install(TARGETS laplacian_hip_fptr DESTINATION bin)
