#include <inttypes.h>
#include <stdlib.h>

/* getopt */
extern int optind;
extern char *optarg;

struct encoder_cl_args
{
    uint32_t width;
    uint32_t height;
    int frame_limit;
    int remote_node;
    char *output_file;
};

struct server_cl_args
{
    int remote_node;
};

typedef struct encoder_cl_args encoder_cl_args_t;
typedef struct server_cl_args server_cl_args_t;

void print_encoder_usage(const char *exec);
void print_server_usage(const char *exec);

encoder_cl_args_t *get_encoder_cl_args(int argc, char **argv);
server_cl_args_t *get_server_cl_args(int argc, char **argv);
