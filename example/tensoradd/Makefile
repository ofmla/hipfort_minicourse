# Define the compilers
FC = hipfc
CC = hipfc
# Define compiler flags
FFLAGS = -Wall -fimplicit-none -O2
CFLAGS = -Wall -O2

# Output binary name
TARGET = main_program

# Object files
OBJ_CPP = c_functions.o
OBJ_MOD = tensor_lib_c.o
OBJ_MAIN = tensoradd_cfun.o

# All target
all: $(TARGET)

# Build the C++ object
$(OBJ_CPP): c_functions.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Build the Fortran module
$(OBJ_MOD): tensor_lib_c.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Build the main Fortran file
$(OBJ_MAIN): tensoradd_cfun.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Link everything into the final executable
$(TARGET): $(OBJ_CPP) $(OBJ_MOD) $(OBJ_MAIN)
	$(FC) $(FFLAGS) -o $(TARGET) $(OBJ_CPP) $(OBJ_MOD) $(OBJ_MAIN)

# Clean target to remove object files and the binary
clean:
	rm -f $(TARGET) $(OBJ_CPP) $(OBJ_MOD) $(OBJ_MAIN)

