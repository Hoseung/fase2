# This sample should be run after make libHEAAN.a file in lib folder

# All Target
all: clean TestHEAAN

# Tool invocations
TestHEAAN:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ test.cpp -o TestHEAAN -L/usr/local/lib -L/usr/local/cuda/lib64 -L../lib/  -lHEAAN -lntl -lgmp -lm -std=c++11 -g -I/../src -lpthread -lcudart
	##-O2 
	@echo 'Finished building target: $@'
	@echo ' '


# Other Targets
clean:
	rm -rf TestHEAAN
