################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src_fpga/acc_ip_funct.cpp \
../src_fpga/dma_test.cpp 

C_SRCS += \
../src_fpga/pcie_utils.c 

OBJS += \
./src_fpga/acc_ip_funct.o \
./src_fpga/dma_test.o \
./src_fpga/pcie_utils.o 

CPP_DEPS += \
./src_fpga/acc_ip_funct.d \
./src_fpga/dma_test.d 

C_DEPS += \
./src_fpga/pcie_utils.d 


# Each subdirectory must supply rules for building sources it contributes
src_fpga/%.o: ../src_fpga/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include -O3 -fPIC -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src_fpga/%.o: ../src_fpga/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I/usr/local/include -L/usr/lib/x86_64-linux-gnu -O3 -fPIC -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


