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

    if (isServer)
    {
        myID = SEGMENT_SERVER;
        otherID = SEGMENT_CLIENT;
        segmentSize = 1 * sizeof(uint8_t) + 50 * sizeof(uint8_t);
    }
    else
    {
        myID = SEGMENT_CLIENT;
        otherID = SEGMENT_SERVER;
        segmentSize = 1 * sizeof(uint8_t) + 50 * sizeof(uint8_t);
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

    struct server_segment *server_ptr = malloc(sizeof(struct server_segment));
    struct client_segment *client_ptr = malloc(sizeof(struct client_segment));

    // map server segment
    int offset = 0;
    if (isServer)
    {
        server_ptr->cmd = local_mapper(localSegment, localMap, &offset, sizeof(uint8_t), &error);
        server_ptr->data = local_mapper(localSegment, localMap, &offset, 50 * sizeof(uint8_t), &error);
    }
    else
    {
        server_ptr->cmd = remote_mapper(remoteSegment, remoteMap, &offset, sizeof(uint8_t), &error);
        server_ptr->data = remote_mapper(remoteSegment, remoteMap, &offset, 50 * sizeof(uint8_t), &error);
    }

    // map client segment
    offset = 0;
    if (isServer)
    {
        client_ptr->cmd = remote_mapper(remoteSegment, remoteMap, &offset, sizeof(uint8_t), &error);
        client_ptr->data = remote_mapper(remoteSegment, remoteMap, &offset, 50 * sizeof(uint8_t), &error);
    }
    else
    {
        client_ptr->cmd = local_mapper(localSegment, localMap, &offset, sizeof(uint8_t), &error);
        client_ptr->data = local_mapper(localSegment, localMap, &offset, 50 * sizeof(uint8_t), &error);
    }

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
