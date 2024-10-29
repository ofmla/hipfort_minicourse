# Compilers
FC = gfortran
CC = gcc
NVCC = nvcc

# Compiler and linker flags
FFLAGS = -Wall -fimplicit-none -O2
CFLAGS = -Wall -O2
NVFLAGS = -O2 -Xcompiler -fPIC

# Output binary name
TARGET = main_program

# Object files
OBJ_F90 = main.o
OBJ_CU = cuda_implementation.o

# All target
all: $(TARGET)

# Build the CUDA object (which may contain C/C++ functions)
$(OBJ_CU): cuda_implementation.cu
	$(NVCC) $(NVFLAGS) -dc $< -o $@  # Compile to an object file

# Build the Fortran main file
$(OBJ_F90): main.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Link everything into the final executable
$(TARGET): $(OBJ_CU) $(OBJ_F90)
	$(FC) $(FFLAGS) -o $(TARGET) $(OBJ_CU) $(OBJ_F90) -L/usr/local/cuda/lib64 -lcudart

# Clean target to remove object files and the binary
clean:
	rm -f $(TARGET) $(OBJ_F90) $(OBJ_CU)
