# Compiler
GPU_TARGET=CDNA4

TARGET=simple_kernel
SOURCES=host.cpp kernel.cpp

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS+= -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD) -O3 -Wall

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(HIPCXX) $(SOURCES) $(HIPFLAGS) $(ICXXFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
