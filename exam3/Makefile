FAST_SRC = $(wildcard fast_codec63/*.c) $(wildcard fast_codec63/*.cu) $(wildcard fast_codec63/*.h)
REFERENCE_SRC = $(wildcard reference_codec63/*.c) $(wildcard reference_codec63/*.h)

YUV_FILE=/mnt/sdcard/cipr/foreman.yuv
COMP_FRAMES?=10
PREV_FRAMES=$(shell cat test/fc.txt)

.PHONY: run_fast
run_fast:
	cd fast_codec63 && ./run.sh --tegra tegra-3

.PHONY: test
test: test/fast_foreman.c63 test/reference_foreman.c63
	@diff test/fast_foreman.c63 test/reference_foreman.c63; \
	if [ $$? -eq 0 ]; then \
            echo "\033[0;32mFast encoder still works\033[0m"; \
    else \
            echo "\033[0;31mYou have broken the encoder!\033[0m"; \
    fi
	@rm test/fast_foreman.c63 test/reference_foreman.c63;

.PHONY: clean_logs
clean_logs:
	rm fast_codec63/logs/*
