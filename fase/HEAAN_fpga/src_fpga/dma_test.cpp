/*
 * dma_test.cpp
 *
 *  Created on: Feb 27, 2021
 *      Author: etri_ai3
 */
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <byteswap.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>
#include <assert.h>
#include <getopt.h>
#include <time.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h> // sleep



#include "pcie_utils.h"

#define MASK_32_LOW        0x00000000FFFFFFFF
#define MASK_32_HIGH       0xFFFFFFFF00000000

#define DDR3_CH0_ADDR      0x00000000
#define DDR3_CH1_ADDR      0x40000000

#define NTT_W_ADDR         0x80000000
#define NTT_DATA_ADDR      0x80080000
#define NTT_DDR_ADDR       0x00000000 //1.
#define INTT_W_ADDR        0x80100000
#define INTT_DATA_ADDR     0x80180000
#define INTT_DDR_ADDR      0x00080000 //2.
#define MODB_A_ADDR        0x80200000
#define MODB_B_ADDR        0x80280000
#define MODB_DDR_ADDR      0x00100000 //3.

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
//dma
#define DMA_SR       0x1004
#define DMA_SA       0x1018
#define DMA_DA       0x1020
#define DMA_BYTE     0x1028

#define DMA_IDLE_MASK 0x00000002

#define N (1<<16)

void dma_reg_test();
//dma_to_device(dev_dma_wt, NTT_DATA_ADDR, 524288, 0, 1, a);  // 65536 * 8 byte = 524288
//dma_from_device(dev_dma_rd, NTT_DDR_ADDR, 524288, 0, 1, a); // 65536 * 8 byte = 524288
void host2mem_rw_test()
{
	uint64_t *a = new uint64_t[N];
	uint64_t *b = new uint64_t[N];
	uint64_t num_bytes;
	uint32_t err=0;
	num_bytes=N*8;

	//char dev_lite[20]   = "/dev/xdma0_user";
	char dev_dma_wt[20] = "/dev/xdma0_h2c_0";
	char dev_dma_rd[20] = "/dev/xdma0_c2h_0";

	for(int i=0;i<N;i++){
		a[i]=(i+1)*100;
		b[i]=0;
	}

	dma_to_device(dev_dma_wt,DDR3_CH1_ADDR, num_bytes, 0, 1, a);
	printf("HOST -> DDR Write Done\n");
	dma_from_device(dev_dma_rd, DDR3_CH1_ADDR, num_bytes, 0, 1, b); // 65536 * 8 byte = 524288
	printf("DDR -> HOST Read Done\n");
	//compare
	for(int i=0;i<N;i++){
		if(a[i]!=b[i]){
			err++;
		}
		if(i%1000==1){
			printf("a[%d]=%016lx, b[%d]=%016lx\n",i,a[i],i,b[i]);
		}
	}
	printf("Total number of errors=%d\n",err);
}
void simple_dma_test(){

	char dev_lite[20]   = "/dev/xdma0_user";
	char dev_dma_wt[20] = "/dev/xdma0_h2c_0";
	char dev_dma_rd[20] = "/dev/xdma0_c2h_0";

	//data generate & transfer
	uint64_t *a = new uint64_t[N];
	uint64_t *b = new uint64_t[N];
	uint32_t num_bytes,num_words;
	uint32_t err=0;
	uint32_t idle=0;
	uint32_t src_addr=0;
	uint32_t dst_addr=0;
	src_addr=DDR3_CH0_ADDR+(1<<8);
	dst_addr=DDR3_CH1_ADDR+(1<<10);

	printf("=== simple dma test start ===\n");
	num_words=1<<13;
	num_bytes=num_words*8;
	for(uint32_t i=0;i<num_words;i++){
		a[i]=(i+1)*1000;
		//b[i]=0xfffffff;
	}
	dma_to_device  (dev_dma_wt, src_addr, num_bytes, 0, 1, a);
	//dma_to_device  (dev_dma_wt, DDR3_CH1_ADDR, num_bytes, 0, 1, b);
	//dma transferDDR3_CH1_ADDR
	//dma on
	printf("1. check IDLE=0\n");
	printf("SR register default=%x\n",reg_read(dev_lite,DMA_SR));
	do{
		idle=reg_read(dev_lite,DMA_SR) & DMA_IDLE_MASK;
	} while(idle==0);
	//dma source address
	printf("2. source address\n");
	reg_write(dev_lite,DMA_SA,src_addr);
	printf("3. destination address\n");
	reg_write(dev_lite,DMA_DA,dst_addr);
	printf("4. number of bytes\n");
	reg_write(dev_lite,DMA_BYTE,num_bytes);
	printf("5. polling IDLE\n");
	do{
		idle=reg_read(dev_lite,DMA_SR)& DMA_IDLE_MASK;
		if(idle==0){ //busy
			sleep(1);
		}
		else { //idle
			printf("idle=%08x\n",idle);
			break;
		}
	}while(1);

	//dma_transfer(dev_lite, DDR3_CH0_ADDR, DDR3_CH1_ADDR, num_bytes);
	//host read
	dma_from_device(dev_dma_rd, dst_addr, num_bytes, 0, 1, b);
	//compare
	for(uint32_t i=0;i<num_words;i++){
		if(a[i]!=b[i]){
			//printf("a[%d]=0x%016lx, b[%d]=0x%016lx\n",i,a[i],i,b[i]);
			err++;
		}
		if(i%1000==1){
			printf("a[%d]=0x%016lx, b[%d]=0x%016lx\n",i,a[i],i,b[i]);
		}
	}
	printf("number of errors=%d\n",err);
	dma_reg_test();
	//dma_on(dev_lite,0);
}
void dma_reg_test(){
	char dev_lite[20]   = "/dev/xdma0_user";
	uint32_t reg_val,addr;
	printf("register read test\n");
	for(int i=0;i<11;i++){
		//addr   = 0x0000+4*i;
		addr   = 0x01000+4*i;
		reg_val=reg_read(dev_lite,addr);
		//reg_val=reg_read(dev_lite,0x80+4*i);
		printf("i=%d: address=0x%08x, reg=0x%08x\n",i,addr,reg_val);
	}

}
void dma_transfer(uint32_t src_addr,uint32_t dst_addr, uint32_t num_bytes){
	uint32_t reg_val;
	char dev_lite[20]   = "/dev/xdma0_user";
	do{
		reg_val=reg_read(dev_lite,DMA_SR) & DMA_IDLE_MASK;
	} while(reg_val==0);
	//dma source address
	//printf("2. source address\n");
	reg_write(dev_lite,DMA_SA,src_addr);
	//printf("3. destination address\n");
	reg_write(dev_lite,DMA_DA,dst_addr);
	//printf("4. number of bytes\n");
	reg_write(dev_lite,DMA_BYTE,num_bytes);
	//printf("5. polling IDLE\n");
	do{
		reg_val=reg_read(dev_lite,DMA_SR)& DMA_IDLE_MASK;
		if(reg_val==0){ //busy
			for(int i=0;i<10;i++){
				//NOP;
			}
		}
		else { //idle
			//printf("idle=%08x\n",idle);
			break;
		}
	}while(1);
}

