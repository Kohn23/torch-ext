CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17 -O2 -Wall
NVCCFLAGS = -std=c++17 -O2 -arch=native

BUILD_DIR = build
TARGET = $(BUILD_DIR)/test_qbmatmul

CU_SRC = qbmatmul.cu
CPP_SRC = tests/test_qbmatmul.cpp

CU_OBJ = $(BUILD_DIR)/qbmatmul.o
CPP_OBJ = $(BUILD_DIR)/test_qbmatmul.o

HEADS = kernel.h qbmatmul.h utils.h


all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): $(CU_OBJ) $(CPP_OBJ)
	$(NVCC) -o $@ $^ $(NVCCFLAGS)

$(CU_OBJ): $(CU_SRC) $(HEADS) | $(BUILD_DIR)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

$(CPP_OBJ): $(CPP_SRC) $(HEADS) | $(BUILD_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS) -I.

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run