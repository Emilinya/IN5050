FAST_SRC = $(wildcard fast_codec63/*.c) $(wildcard fast_codec63/*.h)
REFERENCE_SRC = $(wildcard reference_codec63/*.c) $(wildcard reference_codec63/*.h)

COMP_FRAMES?=10
PREV_FRAMES=$(shell cat test/fc.txt)

.PHONY: test
test: test/fast_foreman.c63 test/reference_foreman.c63
	@diff test/fast_foreman.c63 test/reference_foreman.c63; \
	if [ $$? -eq 0 ]; then \
            echo "\033[0;32mFast decoder still works\033[0m"; \
    else \
            echo "\033[0;31mYou have broken the decoder!\033[0m"; \
    fi
	@rm test/fast_foreman.c63 test/reference_foreman.c63;

test/fast_foreman.c63: fast_codec63/c63enc
	fast_codec63/c63enc -w 352 -h 288 -f $(COMP_FRAMES) -o test/fast_foreman.c63 /mnt/sdcard/cipr/foreman.yuv

test/reference_foreman.c63: reference_codec63/c63enc
	reference_codec63/c63enc -w 352 -h 288 -f $(COMP_FRAMES) -o test/reference_foreman.c63 /mnt/sdcard/cipr/foreman.yuv

fast_codec63/c63enc: $(FAST_SRC)
	make -C fast_codec63/ c63enc

reference_codec63/c63enc: $(REFERENCE_SRC)
	make -C reference_codec63/ c63enc
