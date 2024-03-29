#pragma once

#include <inttypes.h>

extern uint8_t yquanttbl_def[64] __attribute__((aligned(16)));
extern uint8_t uvquanttbl_def[64] __attribute__((aligned(16)));
extern uint16_t DCVLC[2][12];
extern uint8_t DCVLC_Size[2][12];
extern uint8_t DCVLC_num_by_length[2][16];
extern uint8_t DCVLC_data[2][12];
extern uint16_t ACVLC[2][16][11];
extern uint8_t ACVLC_Size[2][16][11];
extern uint8_t ACVLC_num_by_length[2][16];
extern uint8_t ACVLC_data[2][162];
extern uint8_t zigzag_U_table[64];
extern uint8_t zigzag_V_table[64];
extern float dct_lookup_table[8][8];
extern uint16_t MVVLC[8];
extern uint8_t MVVLC_Size[8];
