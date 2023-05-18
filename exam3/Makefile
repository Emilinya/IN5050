YUV_FILE=/opt/Media/foreman.yuv
COMP_FRAMES?=10
PREV_FRAMES=$(shell cat test/fc.txt)

TEGRA=tegra-3
PC=in5050-2016-10

.PHONY: test
test:
	cd fast_codec63 && ./run.sh --tegra $(TEGRA) --args "$(YUV_FILE) -o fast_foreman.c63 -w 352 -h 288 -f $(COMP_FRAMES)"
	rsync -rt $(PC):in5050-codec63-build/x86-build/fast_foreman.c63 test

	@echo
	@echo "### Encoding with reference ###"
	rsync -rt reference_codec63/ $(PC):in5050-reference-codec63
	@echo "Encoding ..." 
	@ssh -t $(PC) "cd in5050-reference-codec63 && make c63enc && ./c63enc $(YUV_FILE) -o reference_foreman.c63 -w 352 -h 288 -f $(COMP_FRAMES)" > /dev/null
	rsync -rt $(PC):in5050-reference-codec63/reference_foreman.c63 test
	
	@diff test/fast_foreman.c63 test/reference_foreman.c63; \
	if [ $$? -eq 0 ]; then \
            echo "\033[0;32mFast encoder still works\033[0m"; \
    else \
            echo "\033[0;31mYou have broken the encoder!\033[0m"; \
    fi
	@rm test/fast_foreman.c63 test/reference_foreman.c63;
	@ssh -t $(PC) "rm in5050-codec63-build/x86-build/fast_foreman.c63 in5050-reference-codec63/reference_foreman.c63;"

.PHONY: time
time:
	cd fast_codec63 && ./run.sh --tegra $(TEGRA) --repeat 10 --args "$(YUV_FILE) -o fast_foreman.c63 -w 352 -h 288 -f 100"

.PHONY: clean_logs
clean_logs:
	rm fast_codec63/logs/*
