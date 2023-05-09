#include <getopt.h>
#include <stdio.h>

#include "cl_utils.h"

void print_encoder_usage(const char *exec)
{
    printf("Usage: %s [options] input_file\n", exec);
    printf("Commandline options:\n");
    printf("  -h                             Height of images to compress\n");
    printf("  -w                             Width of images to compress\n");
    printf("  -o                             Output file (.c63)\n");
    printf("  -r                             Node id of server\n");
    printf("  [-f]                           Limit number of frames to encode\n");
    printf("\n");

    exit(EXIT_FAILURE);
}

void print_server_usage(const char *exec)
{
    printf("Usage: %s -r nodeid\n", exec);
    printf("Commandline options:\n");
    printf("  -r                             Node id of client\n");
    printf("\n");

    exit(EXIT_FAILURE);
}

encoder_cl_args_t *get_encoder_cl_args(int argc, char **argv)
{
    if (argc == 1)
    {
        print_encoder_usage(argv[0]);
    }

    int c;
    encoder_cl_args_t *args = calloc(1, sizeof(encoder_cl_args_t));
    while ((c = getopt(argc, argv, "h:w:o:f:r:i")) != -1)
    {
        switch (c)
        {
        case 'h':
            args->height = atoi(optarg);
            break;
        case 'w':
            args->width = atoi(optarg);
            break;
        case 'o':
            args->output_file = optarg;
            break;
        case 'f':
            args->frame_limit = atoi(optarg);
            break;
        case 'r':
            args->remote_node = atoi(optarg);
            break;
        default:
            print_encoder_usage(argv[0]);
            break;
        }
    }

    if (!args->width)
    {
        fprintf(stderr, "Missing width, see --help.\n");
        exit(EXIT_FAILURE);
    }
    if (!args->height)
    {
        fprintf(stderr, "Missing height, see --help.\n");
        exit(EXIT_FAILURE);
    }
    if (!args->output_file)
    {
        fprintf(stderr, "Missing output file, see --help.\n");
        exit(EXIT_FAILURE);
    }
    if (!args->remote_node)
    {
        fprintf(stderr, "Missing remote node, see --help.\n");
        exit(EXIT_FAILURE);
    }

    if (optind >= argc)
    {
        fprintf(stderr, "Error getting program options, try --help.\n");
        exit(EXIT_FAILURE);
    }

    return args;
}

server_cl_args_t *get_server_cl_args(int argc, char **argv)
{
    if (argc == 1)
    {
        print_server_usage(argv[0]);
    }

    int c;
    server_cl_args_t *args = calloc(1, sizeof(server_cl_args_t));
    while ((c = getopt(argc, argv, "h:w:o:f:i:r:")) != -1)
    {
        switch (c)
        {
        case 'r':
            args->remote_node = atoi(optarg);
            break;
        default:
            print_server_usage(argv[0]);
            break;
        }
    }

    if (!args->remote_node)
    {
        fprintf(stderr, "Missing remote node, see --help.\n");
        exit(EXIT_FAILURE);
    }

    return args;
}
