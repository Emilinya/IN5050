#include "sisci_utils.h"

void *local_mapper(sci_local_segment_t *segment, sci_map_t *map, int *offset, size_t size, sci_error_t *error)
{
    void *pointer = (void *)SCIMapLocalSegment(
        *segment, map, *offset, size, NULL, NO_FLAGS, error);
    ERR_CHECK(*error, "SCIMapRemoteSegment");
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

yuv_t *map_local_yuv(
    sci_local_segment_t *segment, sci_map_t *map, int *offset,
    int ysize, int usize, int vsize, sci_error_t *error)
{
    yuv_t *yuv = (yuv_t *)local_mapper(segment, map, offset, sizeof(yuv_t), error);
    yuv->Y = (uint8_t *)local_mapper(segment, map, offset, ysize * sizeof(uint8_t), error);
    yuv->U = (uint8_t *)local_mapper(segment, map, offset, usize * sizeof(uint8_t), error);
    yuv->V = (uint8_t *)local_mapper(segment, map, offset, vsize * sizeof(uint8_t), error);
    return yuv;
}

dct_t *map_local_dct(
    sci_local_segment_t *segment, sci_map_t *map, int *offset,
    int ysize, int usize, int vsize, sci_error_t *error)
{
    dct_t *dct = (dct_t *)local_mapper(segment, map, offset, sizeof(dct_t), error);
    dct->Ydct = (int16_t *)local_mapper(segment, map, offset, ysize * sizeof(int16_t), error);
    dct->Udct = (int16_t *)local_mapper(segment, map, offset, usize * sizeof(int16_t), error);
    dct->Vdct = (int16_t *)local_mapper(segment, map, offset, vsize * sizeof(int16_t), error);
    return dct;
}

yuv_t *map_remote_yuv(
    sci_remote_segment_t *segment, sci_map_t *map, int *offset,
    int ysize, int usize, int vsize, sci_error_t *error)
{
    yuv_t *yuv = (yuv_t *)remote_mapper(segment, map, offset, sizeof(yuv_t), error);
    yuv->Y = (uint8_t *)remote_mapper(segment, map, offset, ysize * sizeof(uint8_t), error);
    yuv->U = (uint8_t *)remote_mapper(segment, map, offset, usize * sizeof(uint8_t), error);
    yuv->V = (uint8_t *)remote_mapper(segment, map, offset, vsize * sizeof(uint8_t), error);
    return yuv;
}

dct_t *map_remote_dct(
    sci_remote_segment_t *segment, sci_map_t *map, int *offset,
    int ysize, int usize, int vsize, sci_error_t *error)
{
    dct_t *dct = (dct_t *)remote_mapper(segment, map, offset, sizeof(dct_t), error);
    dct->Ydct = (int16_t *)remote_mapper(segment, map, offset, ysize * sizeof(int16_t), error);
    dct->Udct = (int16_t *)remote_mapper(segment, map, offset, usize * sizeof(int16_t), error);
    dct->Vdct = (int16_t *)remote_mapper(segment, map, offset, vsize * sizeof(int16_t), error);
    return dct;
}

void sisci_init(
    int initServer, int localAdapterNo, int remoteNodeId, sci_desc_t *sd, sci_map_t *localMap,
    sci_map_t *remoteMap, sci_local_segment_t *localSegment, sci_remote_segment_t *remoteSegment,
    volatile struct server_segment **server_segment, volatile struct client_segment **client_segment,
    struct c63_common *cm)
{
    int myID;
    int otherID;
    int segmentSize;

    int ysize = cm->ypw * cm->yph;
    int usize = cm->upw * cm->uph;
    int vsize = cm->vpw * cm->vph;
    int imsize = ysize + usize + vsize;
    int mb_count = cm->mb_rows * cm->mb_cols;

    if (initServer) {
        myID = SEGMENT_SERVER;
        otherID = SEGMENT_CLIENT;
        segmentSize = sizeof(struct server_segment) + 3 * sizeof(yuv_t) + sizeof(dct_t);
        segmentSize += imsize * (3 * sizeof(uint8_t) + sizeof(int16_t));
        segmentSize += (mb_count + mb_count / 2) * sizeof(struct macroblock);
    } else {
        myID = SEGMENT_CLIENT;
        otherID = SEGMENT_SERVER;
        segmentSize = sizeof(struct client_segment) + sizeof(yuv_t) + imsize * sizeof(uint8_t);
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

    SCIPrepareSegment(*localSegment, localAdapterNo, NO_FLAGS, &error);
    ERR_CHECK(error, "SCIPrepareSegment");

    SCISetSegmentAvailable(*localSegment, localAdapterNo, NO_FLAGS, &error);
    ERR_CHECK(error, "SCISetSegmentAvailable");

    while (1) {
        SCIConnectSegment(
            *sd, remoteSegment, remoteNodeId, otherID, localAdapterNo,
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

    // map server segment
    int offset = 0;
    if (initServer) {
        *server_segment = local_mapper(
            localSegment, localMap, &offset, sizeof(struct server_segment), &error);
        (*server_segment)->reference_recons = map_local_yuv(
            localSegment, localMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->currenct_recons = map_local_yuv(
            localSegment, localMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->predicted = map_local_yuv(
            localSegment, localMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->residuals = map_local_dct(
            localSegment, localMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->mbs[Y_COMPONENT] = local_mapper(
            localSegment, localMap, &offset, mb_count * sizeof(struct macroblock), &error);
        (*server_segment)->mbs[U_COMPONENT] = local_mapper(
            localSegment, localMap, &offset, mb_count / 4 * sizeof(struct macroblock), &error);
        (*server_segment)->mbs[V_COMPONENT] = local_mapper(
            localSegment, localMap, &offset, mb_count / 4 * sizeof(struct macroblock), &error);
    } else {
        *server_segment = remote_mapper(
            remoteSegment, remoteMap, &offset, sizeof(struct server_segment), &error);
        (*server_segment)->reference_recons = map_remote_yuv(
            remoteSegment, remoteMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->currenct_recons = map_remote_yuv(
            remoteSegment, remoteMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->predicted = map_remote_yuv(
            remoteSegment, remoteMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->residuals = map_remote_dct(
            remoteSegment, remoteMap, &offset, ysize, usize, vsize, &error);
        (*server_segment)->mbs[Y_COMPONENT] = remote_mapper(
            remoteSegment, remoteMap, &offset, mb_count * sizeof(struct macroblock), &error);
        (*server_segment)->mbs[U_COMPONENT] = remote_mapper(
            remoteSegment, remoteMap, &offset, mb_count / 4 * sizeof(struct macroblock), &error);
        (*server_segment)->mbs[V_COMPONENT] = remote_mapper(
            remoteSegment, remoteMap, &offset, mb_count / 4 * sizeof(struct macroblock), &error);
    }

    // map client segment
    offset = 0;
    if (initServer) {
        *client_segment = remote_mapper(remoteSegment, remoteMap, &offset, sizeof(struct client_segment), &error);
        (*client_segment)->image = map_remote_yuv(
            remoteSegment, remoteMap, &offset, ysize, usize, vsize, &error);
    } else {
        *client_segment = local_mapper(localSegment, localMap, &offset, sizeof(struct client_segment), &error);
        (*client_segment)->image = map_local_yuv(
            localSegment, localMap, &offset, ysize, usize, vsize, &error);
    }
}
