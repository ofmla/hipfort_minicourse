.PHONY: build run clean

build: main

main:
	$(FC) $(CFLAGS) main.f90 hip_implementation.cpp -I/opt/rocm/include -o main 

run: main
	./main

clean:
	rm -f main *.o *.mod
