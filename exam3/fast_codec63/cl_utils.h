#include <inttypes.h>
#include <stdlib.h>

/* getopt */
extern int optind;
extern char *optarg;

struct cl_args
{
    uint32_t width;
    uint32_t height;
    int frame_limit;
    int remote_node;
    char *output_file;
};

typedef struct cl_args cl_args_t;

void print_usage(const char *exec);

cl_args_t *get_cl_args(int argc, char **argv);
