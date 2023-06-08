################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/BootContext.cpp \
../src/Ciphertext.cpp \
../src/EvaluatorUtils.cpp \
../src/HEAAN.cpp \
../src/Key.cpp \
../src/Plaintext.cpp \
../src/Ring.cpp \
../src/RingMultiplier.cpp \
../src/Scheme.cpp \
../src/SchemeAlgo.cpp \
../src/SecretKey.cpp \
../src/SerializationUtils.cpp \
../src/StringUtils.cpp \
../src/TestScheme.cpp \
../src/TimeUtils.cpp \
../src/cu_ring_mult.cu \
../src/cu_crt.cu \
../src/cu_icrt.cu \
../src/cu_mult_handle.cu


OBJS += \
./src/BootContext.o \
./src/Ciphertext.o \
./src/EvaluatorUtils.o \
./src/HEAAN.o \
./src/Key.o \
./src/Plaintext.o \
./src/Ring.o \
./src/RingMultiplier.o \
./src/Scheme.o \
./src/SchemeAlgo.o \
./src/SecretKey.o \
./src/SerializationUtils.o \
./src/StringUtils.o \
./src/TestScheme.o \
./src/TimeUtils.o \
./src/cu_ring_mult.o \
./src/cu_crt.o \
./src/cu_icrt.o \
./src/cu_mult_handle.o 


CPP_DEPS += \
./src/BootContext.d \
./src/Ciphertext.d \
./src/EvaluatorUtils.d \
./src/HEAAN.d \
./src/Key.d \
./src/Plaintext.d \
./src/Ring.d \
./src/RingMultiplier.d \
./src/Scheme.d \
./src/SchemeAlgo.d \
./src/SecretKey.d \
./src/SerializationUtils.d \
./src/StringUtils.d \
./src/TestScheme.d \
./src/TimeUtils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -fPIC -I/usr/local/include -L/usr/local/lib -O3 -c -std=c++11 -pthread -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
#	nvcc -Xcompiler -fPIC  -I/home/etri_ai2/work/fhe-ai-sw-etri/third_party/CGBN/include -c -o "$@" "$<"  -arch=sm_75 -rdc=true
	nvcc -Xcompiler -fPIC  -I$(CGBN_INCLUDE_DIR) -c -o "$@" "$<"  -arch=sm_75 -rdc=true
	@echo 'Finished building: $<'
	@echo ' '

#nvcc -c -o "$@" "$<"  -gencode arch=compute_70,code=compute_70
