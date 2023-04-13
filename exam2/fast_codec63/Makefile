NVCC=nvcc

CPUARCH=-march=armv8.2-a
GPUARCH=-arch=compute_72 -code=compute_72

CFLAGS = -O1 -g
LDFLAGS = -lm $(GPUARCH) -Xcompiler $(CPUARCH) # -fsanitize=address # -lnvToolsExt
CCFLAGS = $(CFLAGS) $(CPUARCH) -fno-tree-vectorize -Wall -Wextra # -fsanitize=address
CUFLAGS = $(CFLAGS) $(GPUARCH) -G -rdc=true -Xcompiler $(CPUARCH) # -fsanitize=address

# Use the compiler to generate make rules
MFLAGS = -MMD -MP -MF

SOURCES = $(wildcard *.c) $(wildcard *.cu)

MAIN_FILES = c63enc.c.o c63dec.c.o c63pred.c.o
OBJECTS = $(filter-out $(addprefix build/,$(MAIN_FILES)), $(patsubst %,build/%.o,$(SOURCES)))

DEPENDENCIES = $(patsubst %,build/%.d,$(SOURCES))


.PHONY: all
all: c63enc c63dec c63pred

c63enc: build/c63enc.c.o $(OBJECTS)
	$(NVCC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: build/c63dec.c.o $(OBJECTS)
	$(NVCC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: build/c63dec.c.o $(OBJECTS)
	$(NVCC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

build/%.cu.o: %.cu
	@mkdir -p build
	$(NVCC) $(CUFLAGS) $(MFLAGS) $(patsubst build/%.cu.o,build/.%.cu.d,$@) -c $< -o $@

build/%.c.o: %.c
	@mkdir -p build
	$(CC) $(CCFLAGS) $(MFLAGS) $(patsubst build/%.c.o,build/.%.c.d,$@) -c $< -o $@

.PHONY: clean
clean:
	$(RM) c63enc c63dec c63pred
	$(RM) -r build

-include $(DEPENDENCIES)
