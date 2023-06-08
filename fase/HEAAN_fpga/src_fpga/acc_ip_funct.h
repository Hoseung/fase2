/*
 * acc_ip_funct.h
 *
 *  Created on: Mar 4, 2021
 *      Author: etri_ai3
 */

#ifndef ACC_IP_FUNCT_H_
#define ACC_IP_FUNCT_H_
#include <stdint.h>
#include "pcie_utils.h"

#define MASK_32_LOW        0x00000000FFFFFFFF
#define MASK_32_HIGH       0xFFFFFFFF00000000

#define DDR3_CH0_ADDR      0x00000000
#define DDR3_CH1_ADDR      0x40000000

#define NTT_W_ADDR         0x80000000
#define NTT_DATA_ADDR      0x80080000

#define INTT_W_ADDR        0x80100000
#define INTT_DATA_ADDR     0x80180000

#define MODB_A_ADDR        0x80200000
#define MODB_B_ADDR        0x80280000

#define NTT_DDR_ADDR       DDR3_CH0_ADDR //1. NTT OUTPUT DDR Address
#define INTT_DDR_ADDR      NTT_DDR_ADDR+((1<<16)*8)*64 //2. INTT OUTPUT DDR Address
#define MODB_DDR_ADDR      INTT_DDR_ADDR+((1<<16)*8)*64 //3. MODB OUTPUT DDR Address

// PCIe AXI-Lite Base addr: 0x9000_0000
#define CSR_NTT_P_L        0x00
#define CSR_NTT_P_H        0x04
#define CSR_NTT_PINV_L     0x08
#define CSR_NTT_PINV_H     0x0C
#define CSR_NTT_DDR        0x10
#define CSR_NTT_W_STAT     0x18
#define CSR_NTT_DONE       0x20
#define CSR_NTT_IN_CNT     0x28
#define CSR_NTT_OUT_CNT    0x30

#define CSR_INTT_P_L       0x40
#define CSR_INTT_P_H       0x44
#define CSR_INTT_PINV_L    0x48
#define CSR_INTT_PINV_H    0x4C
#define CSR_INTT_NSCALE_L  0x50
#define CSR_INTT_NSCALE_H  0x54
#define CSR_INTT_DDR       0x58
#define CSR_INTT_W_STAT    0x60
#define CSR_INTT_DONE      0x68
#define CSR_INTT_IN_CNT    0x70
#define CSR_INTT_OUT_CNT   0x78

#define CSR_MODB_P_L       0x80
#define CSR_MODB_P_H       0x84
#define CSR_MODB_PR_L      0x88
#define CSR_MODB_PR_H      0x8C
#define CSR_MODB_KBAR2_L   0x90
#define CSR_MODB_KBAR2_H   0x94
#define CSR_MODB_DDR       0x98
#define CSR_MODB_STAT      0xA0
#define CSR_MODB_DONE      0xA8
#define CSR_MODB_IN_CNT    0xB0
#define CSR_MODB_OUT_CNT   0xB8


void NTT_FPGA(uint64_t* a, uint64_t *scaledRootPows, uint64_t pVec, uint64_t pInvVec);
void NTT_MODE_FPGA(uint64_t* a, uint64_t *scaledRootPows, uint64_t pVec, uint64_t pInvVec,uint8_t mode);
void INTT_FPGA(uint64_t* a, uint64_t *scaledRootInvPows, uint64_t pVec, uint64_t pInvVec, uint64_t scaledNInv);
void MODB_FPGA(uint64_t *rxi, uint64_t *rai, uint64_t *rbi, uint64_t pr, uint64_t pri,uint64_t kbar2);
double estimate_time(struct timespec *begin, struct timespec *end);
void MODB_FPGA_DMA(uint64_t *rxi, uint64_t *rai, uint64_t *rbi, uint32_t a_addr, uint32_t b_addr,uint32_t out_addr,uint64_t pr, uint64_t pri,uint64_t kbar2);



#endif /* ACC_IP_FUNCT_H_ */
