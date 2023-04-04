# Acknowledgement: Functionality for creating make rules of dependencies is
# based on code presented here <http://codereview.stackexchange.com/q/11109>

CFLAGS = -O1 -fno-tree-vectorize -Wall -Wextra -g -pg -march=armv8.2-a
LDFLAGS = -lm
# Use the compiler to generate make rules
MFLAGS = -MMD -MP -MF

SOURCES = $(wildcard *.c)

MAIN_FILES = c63enc.o c63dec.o c63pred.o
OBJECTS = $(filter-out $(addprefix build/,$(MAIN_FILES)), $(addprefix build/,$(SOURCES:.c=.o)))

DEPENDENCIES = $(addprefix build/.,$(SOURCES:.c=.d))


.PHONY: all
all: c63enc c63dec c63pred

c63enc: build/c63enc.o $(OBJECTS)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: build/c63dec.o $(OBJECTS)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: build/c63dec.o $(OBJECTS)
	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

build/%.o: %.c
	@mkdir -p build
	$(CC) $(CFLAGS) $(MFLAGS) $(patsubst build/%,build/.%,$(patsubst %.o,%.d,$@)) -c $< -o $@

.PHONY: clean
clean:
	$(RM) c63enc c63dec c63pred build/*

-include $(DEPENDENCIES)
