# Define compilers
FC = gfortran               # Fortran compiler for CPU-only program
HIPFC = /usr/local/hipfort/bin/hipfc               # HIP Fortran compiler for GPU program

# Define source files
CPU_SRC = main_cpu.f90
GPU_SRC = main_hipfort.f90 kernel.cpp

# Define output executables
CPU_EXEC = cpu_exec
GPU_EXEC = gpu_exec

# Compiler flags
FCFLAGS = -O2               # Optimization for CPU
HIPFCFLAGS = -O2 -I/opt/rocm/include # Add include path for ROCm headers

# Targets
all: $(CPU_EXEC) $(GPU_EXEC)

# CPU-only executable
$(CPU_EXEC): $(CPU_SRC)
    $(FC) $(FCFLAGS) -o $@ $^

# GPU-enabled executable
$(GPU_EXEC): $(GPU_SRC)
    $(HIPFC) $(HIPFCFLAGS) -o $@ $^

# Clean up build artifacts
clean:
    rm -f $(CPU_EXEC) $(GPU_EXEC) *.o *.mod
