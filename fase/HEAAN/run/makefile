# This sample should be run after make libHEAAN.a file in lib folder

# All Target
all: clean TestHEAAN

# Tool invocations
#
SimpleTest:
	g++ test1.cpp -o simpleTest -L/usr/local/lib ../lib/libHEAAN.a -Wl,-rpath,/usr/local/lib -lntl -lgmp -lm -std=c++11 -g -I/../src -pthread 

TestHEAAN:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ test.cpp -o TestHEAAN -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -g -I/../src -pthread 
	##-O2 
	@echo 'Finished building target: $@'
	@echo ' '

Key:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ load_key.cpp -o LoadKey -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -g -I/../src -pthread 
	##-O2 
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	rm -rf TestHEAAN
