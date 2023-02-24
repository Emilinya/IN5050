FAST_SRC = $(wildcard fast_codec63/*.c) $(wildcard fast_codec63/*.h)
REFERENCE_SRC = $(wildcard reference_codec63/*.c) $(wildcard reference_codec63/*.h)

COMP_FRAMES?=10
PREV_FRAMES=$(shell cat test/fc.txt)

.PHONY: test
test: test/fast_foreman.c63 test/reference_foreman.c63
	@if [ "$(PREV_FRAMES)" != "$(COMP_FRAMES)" ]; then \
			echo "Frame count has changed, remaking files"; \
			rm test/fast_foreman.c63 test/reference_foreman.c63; \
			make test/fast_foreman.c63; \
			make test/reference_foreman.c63; \
			echo $(COMP_FRAMES) > test/fc.txt; \
    fi
	@diff test/fast_foreman.c63 test/reference_foreman.c63; \
	if [ $$? -eq 0 ]; then \
            echo "\033[0;32mFast decoder still works\033[0m"; \
    else \
            echo "\033[0;31mYou have broken the decoder!\033[0m"; \
    fi
	@rm test/fast_foreman.c63 test/reference_foreman.c63;

.PHONY: profile_reference
profile_reference: reference_codec63/c63enc
	reference_codec63/c63enc -w 352 -h 288 -f 100 -o profiled_reference_foreman.c63 foreman.yuv
	gprof reference_codec63/c63enc gmon.out > profiling/reference_profile.txt; \
	rm profiled_reference_foreman.c63
	rm gmon.out

.PHONY: profile_fast
profile_fast: fast_codec63/c63enc
	fast_codec63/c63enc -w 352 -h 288 -f 100 -o profiled_fast_foreman.c63 -p 10 foreman.yuv
	gprof fast_codec63/c63enc gmon.out > profiling/fast_profile.txt; \
	rm profiled_fast_foreman.c63
	rm gmon.out

test/fast_foreman.c63: fast_codec63/c63enc
	fast_codec63/c63enc -w 352 -h 288 -f $(COMP_FRAMES) -o test/fast_foreman.c63 foreman.yuv

test/reference_foreman.c63: reference_codec63/c63enc
	reference_codec63/c63enc -w 352 -h 288 -f $(COMP_FRAMES) -o test/reference_foreman.c63 foreman.yuv

fast_codec63/c63enc: $(FAST_SRC)
	make -C fast_codec63/ c63enc

reference_codec63/c63enc: $(REFERENCE_SRC)
	make -C reference_codec63/ c63enc
