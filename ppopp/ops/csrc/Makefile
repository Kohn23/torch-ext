NVCC = nvcc
# Suppress deprecation warnings for Thrust and CUB when compiling with newer CUDA versions
NVCC_FLAGS = -std=c++17 -O3 -arch=native -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
INCLUDES = -I. -I./common
LDLIBS = -lcublas -lcusolver -lcurand

OBJDIR = build
TESTDIR = tests

ifeq ($(OS),Windows_NT)
    OBJEXT = .obj
else
    OBJEXT = .o
    
endif

RM_CMD   = rm -rf
MKDIR_P  = mkdir -p


SRCS := qbmatmul.cu $(wildcard $(TESTDIR)/*.cu) $(filter-out common/tests/%,$(wildcard common/**/*.cu))

OBJECTS := $(patsubst %.cu,$(OBJDIR)/%$(OBJEXT),$(SRCS))

DEPS := $(addsuffix .d,$(OBJECTS))
-include $(DEPS)

TARGET = $(OBJDIR)/test_qbfactorization

all: $(OBJDIR) $(TARGET)

$(OBJDIR):
	@$(MKDIR_P) $(OBJDIR)

$(TARGET): $(OBJECTS)
	$(NVCC) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%$(OBJEXT): %.cu | $(OBJDIR)
	@$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	@$(RM_CMD) $(OBJDIR)

.PHONY: all clean