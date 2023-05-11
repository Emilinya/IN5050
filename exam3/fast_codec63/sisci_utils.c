#include "sisci_utils.h"

void *local_mapper(sci_local_segment_t *segment, sci_map_t *map, int *offset, size_t size, sci_error_t *error)
{
    void *pointer = (void *)SCIMapLocalSegment(
        *segment, map, *offset, size, NULL, NO_FLAGS, error);
    ERR_CHECK(*error, "SCIMapLocalSegment");
    *offset += size;
    return pointer;
}

void *remote_mapper(sci_remote_segment_t *segment, sci_map_t *map, int *offset, size_t size, sci_error_t *error)
{
    void *pointer = (void *)SCIMapRemoteSegment(
        *segment, map, *offset, size, NULL, NO_FLAGS, error);
    ERR_CHECK(*error, "SCIMapRemoteSegment");
    *offset += size;
    return pointer;
}

void *mapper(
    int isRemote, sci_local_segment_t *localSegment, sci_remote_segment_t *remoteSegment,
    sci_map_t *localMap, sci_map_t *remoteMap, int *offset, size_t size, sci_error_t *error)
{
    if (isRemote) {
        return remote_mapper(remoteSegment, remoteMap, offset, size, error);
    } else {
        return local_mapper(localSegment, localMap, offset, size, error);
    }
}

void map_yuv(
    yuv_t *yuv, int isRemote, sci_local_segment_t *ls, sci_remote_segment_t *rs, sci_map_t *lm,
    sci_map_t *rm, int *offset, size_t ysize, size_t usize, size_t vsize, sci_error_t *error)
{
    yuv->Y = mapper(isRemote, ls, rs, lm, rm, offset, ysize * sizeof(uint8_t), error);
    yuv->U = mapper(isRemote, ls, rs, lm, rm, offset, usize * sizeof(uint8_t), error);
    yuv->V = mapper(isRemote, ls, rs, lm, rm, offset, vsize * sizeof(uint8_t), error);
}

void map_dct(
    dct_t *yuv, int isRemote, sci_local_segment_t *ls, sci_remote_segment_t *rs, sci_map_t *lm,
    sci_map_t *rm, int *offset, size_t ysize, size_t usize, size_t vsize, sci_error_t *error)
{
    yuv->Ydct = mapper(isRemote, ls, rs, lm, rm, offset, ysize * sizeof(int16_t), error);
    yuv->Udct = mapper(isRemote, ls, rs, lm, rm, offset, usize * sizeof(int16_t), error);
    yuv->Vdct = mapper(isRemote, ls, rs, lm, rm, offset, vsize * sizeof(int16_t), error);
}

void sisci_init(
    int isServer, int remoteNodeId, sci_desc_t *sd, sci_map_t *localMap, sci_map_t *remoteMap,
    sci_local_segment_t *localSegment, sci_remote_segment_t *remoteSegment,
    struct server_segment **server_segment, struct client_segment **client_segment, struct c63_common *cm)
{
    int myID;
    int otherID;
    int segmentSize;

    int ysize = cm->ypw * cm->yph;
    int usize = cm->upw * cm->uph;
    int vsize = cm->vpw * cm->vph;
    int imsize = ysize + usize + vsize;
    int mb_count = cm->mb_rows * cm->mb_cols;
    size_t mb_size = sizeof(struct macroblock);

    if (isServer)
    {
        myID = SEGMENT_SERVER;
        otherID = SEGMENT_CLIENT;
        segmentSize = (1 + 3 * imsize) * sizeof(uint8_t) + imsize * sizeof(int16_t);
        segmentSize += (mb_count + mb_count / 2) * mb_size;
    }
    else
    {
        myID = SEGMENT_CLIENT;
        otherID = SEGMENT_SERVER;
        segmentSize = (1 + imsize) * sizeof(uint8_t);
    }

    sci_error_t error;
    SCIInitialize(NO_FLAGS, &error);
    ERR_CHECK(error, "SCIInitialize");

    SCIOpen(sd, NO_FLAGS, &error);
    ERR_CHECK(error, "SCIOpen");

    SCICreateSegment(
        *sd, localSegment, myID, segmentSize,
        NO_CALLBACK, NULL, NO_FLAGS, &error);
    ERR_CHECK(error, "SCICreateSegment");

    SCIPrepareSegment(*localSegment, LOCAL_ADAPTER_NUMBER, NO_FLAGS, &error);
    ERR_CHECK(error, "SCIPrepareSegment");

    SCISetSegmentAvailable(*localSegment, LOCAL_ADAPTER_NUMBER, NO_FLAGS, &error);
    ERR_CHECK(error, "SCISetSegmentAvailable");

    while (1)
    {
        SCIConnectSegment(
            *sd, remoteSegment, remoteNodeId, otherID, LOCAL_ADAPTER_NUMBER,
            NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        if (error != SCI_ERR_NO_SUCH_SEGMENT)
        {
            ERR_CHECK(error, "SCIConnectSegment");
        }
        if (error == SCI_ERR_OK)
        {
            break;
        }
    }

    // create shorthand definitions
    sci_remote_segment_t *rs = remoteSegment;
    sci_local_segment_t *ls = localSegment;
    sci_map_t *rm = remoteMap;
    sci_map_t *lm = localMap;

    // create segment structs
    struct server_segment *server_ptr = malloc(sizeof(struct server_segment));
    server_ptr->reference_recons = malloc(sizeof(yuv_t));
    server_ptr->currenct_recons = malloc(sizeof(yuv_t));
    server_ptr->predicted = malloc(sizeof(yuv_t));
    server_ptr->residuals = malloc(sizeof(dct_t));

    struct client_segment *client_ptr = malloc(sizeof(struct client_segment));
    client_ptr->image = malloc(sizeof(yuv_t));

    // map server segment
    int offset = 0;
    server_ptr->cmd = mapper(!isServer, ls, rs, lm, rm, &offset, sizeof(uint8_t), &error);
    map_yuv(server_ptr->reference_recons, !isServer, ls, rs, lm, rm, &offset, ysize, usize, vsize, &error);
    map_yuv(server_ptr->currenct_recons, !isServer, ls, rs, lm, rm, &offset, ysize, usize, vsize, &error);
    map_yuv(server_ptr->predicted, !isServer, ls, rs, lm, rm, &offset, ysize, usize, vsize, &error);
    map_dct(server_ptr->residuals, !isServer, ls, rs, lm, rm, &offset, ysize, usize, vsize, &error);

    server_ptr->mbs[Y_COMPONENT] = mapper(!isServer, ls, rs, lm, rm, &offset, mb_count * mb_size, &error);
    server_ptr->mbs[U_COMPONENT] = mapper(!isServer, ls, rs, lm, rm, &offset, mb_count / 4 * mb_size, &error);
    server_ptr->mbs[V_COMPONENT] = mapper(!isServer, ls, rs, lm, rm, &offset, mb_count / 4 * mb_size, &error);

    // map client segment
    offset = 0;
    client_ptr->cmd = mapper(isServer, ls, rs, lm, rm, &offset, sizeof(uint8_t), &error);
    map_yuv(client_ptr->image, isServer, ls, rs, lm, rm, &offset, ysize, usize, vsize, &error);

    // return segments
    (*server_segment) = server_ptr;
    (*client_segment) = client_ptr;
}

void sisci_create_interrupt(
    int isServer, int remoteNodeId, sci_desc_t *sd, sci_local_interrupt_t *localInterrupt,
    sci_remote_interrupt_t *remoteInterrupt)
{
    unsigned int my_interrupt_number, other_interrupt_number;

    if (isServer)
    {
        my_interrupt_number = SERVER_INTERRUPT_NUMBER;
        other_interrupt_number = CLIENT_INTERRUPT_NUMBER;
    }
    else
    {
        my_interrupt_number = CLIENT_INTERRUPT_NUMBER;
        other_interrupt_number = SERVER_INTERRUPT_NUMBER;
    }

    sci_error_t error;
    SCICreateInterrupt(
        *sd, localInterrupt, LOCAL_ADAPTER_NUMBER, &my_interrupt_number,
        NULL, NULL, SCI_FLAG_FIXED_INTNO, &error);
    ERR_CHECK(error, "SCICreateInterrupt");

    while (1)
    {
        SCIConnectInterrupt(
            *sd, remoteInterrupt, remoteNodeId, LOCAL_ADAPTER_NUMBER,
            other_interrupt_number, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        if (error == SCI_ERR_NO_SUCH_INTNO)
        {
            continue;
        }
        else if (error == SCI_ERR_OK)
        {
            break;
        }
        else
        {
            ERR_CHECK(error, "SCIConnectInterrupt");
        }
    }
}
