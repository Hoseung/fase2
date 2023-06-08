#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include "acc_ip_funct.h"
#include "dma_test.h"
//#include "parameter.h"


char dev_lite[20]   = "/dev/xdma0_user";
char dev_dma_wt[20] = "/dev/xdma0_h2c_0";
char dev_dma_rd[20] = "/dev/xdma0_c2h_0";

int debug1 = 0;
int debug2 = 1;


void NTT_FPGA(uint64_t* a, uint64_t *scaledRootPows, uint64_t pVec, uint64_t pInvVec) {
	uint64_t p = pVec;
	uint64_t pInv = pInvVec;
	int data_i=0;
	uint32_t check_done=1;
    int check_cnt=0;
	//software reset
	//reg_write(dev_lite,0xc0,1);
	//nop
	//for(int i=0;i<1000;i++) {
	//}

	//=== Write parameter
	uint32_t wpl = (uint32_t) (MASK_32_LOW & p); 
	uint32_t wph = (uint32_t) ((p & MASK_32_HIGH) >> 32);
    reg_write(dev_lite, CSR_NTT_P_L, wpl);
    reg_write(dev_lite, CSR_NTT_P_H, wph);

	wpl = (uint32_t) (MASK_32_LOW & pInv);
	wph = (uint32_t) ((pInv & MASK_32_HIGH) >>32);
    reg_write(dev_lite, CSR_NTT_PINV_L, wpl);
    reg_write(dev_lite, CSR_NTT_PINV_H, wph);
    reg_write(dev_lite, CSR_NTT_DDR, NTT_DDR_ADDR);

	//== Write weight
        dma_to_device(dev_dma_wt, NTT_W_ADDR , 524280, 0, 1, scaledRootPows); // 65535 x 8 byte = 524280
	//== Write NTT data
        dma_to_device(dev_dma_wt, NTT_DATA_ADDR, 524288, 0, 1, a);  // 65536 * 8 byte = 524288

	// check NTT Done
	do {
	        check_done = reg_read(dev_lite, CSR_NTT_DONE);
		if(check_done != 0) {
        		reg_write(dev_lite, CSR_NTT_DONE, 0);  	
			break; 
		}

	} while(1);

	// read data
        dma_from_device(dev_dma_rd, NTT_DDR_ADDR, 524288, 0, 1, a); // 65536 * 8 byte = 524288
}

void NTT_MODE_FPGA(uint64_t* a, uint64_t *scaledRootPows, uint64_t pVec, uint64_t pInvVec,uint8_t mode) {
	uint64_t p = pVec;
	uint64_t pInv = pInvVec;
	int data_i=0;
	uint32_t check_done=1;
        int check_cnt=0;
	//software reset
	//reg_write(dev_lite,0xc0,1);
	//nop
	//for(int i=0;i<1000;i++) {
	//}

	//=== Write parameter
	uint32_t wpl = (uint32_t) (MASK_32_LOW & p); 
	uint32_t wph = (uint32_t) ((p & MASK_32_HIGH) >> 32);
	if(mode==0) {
        	reg_write(dev_lite, CSR_NTT_P_L, wpl);  
        	reg_write(dev_lite, CSR_NTT_P_H, wph);  
	}
	wpl = (uint32_t) (MASK_32_LOW & pInv);
	wph = (uint32_t) ((pInv & MASK_32_HIGH) >>32);
	if(mode==0){
		reg_write(dev_lite, CSR_NTT_PINV_L, wpl);  
        	reg_write(dev_lite, CSR_NTT_PINV_H, wph);  
        	reg_write(dev_lite, CSR_NTT_DDR, NTT_DDR_ADDR); 
	}
	//== Write weight
	if(mode==0){
        	dma_to_device(dev_dma_wt, NTT_W_ADDR , 524280, 0, 1, scaledRootPows); // 65535 x 8 byte = 524280
	}
	//== Write NTT data
        dma_to_device(dev_dma_wt, NTT_DATA_ADDR, 524288, 0, 1, a);  // 65536 * 8 byte = 524288

	// check NTT Done
	do {
	        check_done = reg_read(dev_lite, CSR_NTT_DONE);
		if(check_done != 0) {
        		reg_write(dev_lite, CSR_NTT_DONE, 0);  	
			break; 
		}

	} while(1);

	// read data
        dma_from_device(dev_dma_rd, NTT_DDR_ADDR, 524288, 0, 1, a); // 65536 * 8 byte = 524288
}


void INTT_FPGA(uint64_t* a, uint64_t *scaledRootInvPows, uint64_t pVec, uint64_t pInvVec, uint64_t scaledNInv) {
	uint64_t p = pVec;
	uint64_t pInv = pInvVec;
   if (debug1) {printf("## Debug: function(%s()) line(%d)\n", __FUNCTION__, __LINE__); fflush(stdout);}

	//uint64_t wData[65536];
	int data_i=0;
	uint32_t check_done=1;

	//=== Write parameter
	uint32_t wpl = (uint32_t) (MASK_32_LOW & p);          
	uint32_t wph = (uint32_t) ((p & MASK_32_HIGH) >> 32); 
        reg_write(dev_lite, CSR_INTT_P_L, wpl);      
        reg_write(dev_lite, CSR_INTT_P_H, wph);      
	wpl = (uint32_t) (MASK_32_LOW & pInv);                
	wph = (uint32_t) ((pInv & MASK_32_HIGH) >>32);        
        reg_write(dev_lite, CSR_INTT_PINV_L, wpl);    
        reg_write(dev_lite, CSR_INTT_PINV_H, wph);    
        reg_write(dev_lite, CSR_INTT_DDR, INTT_DDR_ADDR); 

	uint64_t NScale = scaledNInv;
	//=== Write parameter
	wpl = (uint32_t) (MASK_32_LOW & NScale);             
	wph = (uint32_t) ((NScale & MASK_32_HIGH) >>32);     
        reg_write(dev_lite, CSR_INTT_NSCALE_L, wpl);  
        reg_write(dev_lite, CSR_INTT_NSCALE_H, wph);  

	//== Write weight
        dma_to_device(dev_dma_wt, INTT_W_ADDR, 524280, 0, 1, scaledRootInvPows);  // 65535 x 8 byte = 524280
	//== Write NTT data
        dma_to_device(dev_dma_wt, INTT_DATA_ADDR, 524288, 0, 1, a);  // 65536 * 8 byte = 524288

	// check INTT Done
	do {
	        check_done = reg_read(dev_lite, CSR_INTT_DONE);
		if(check_done != 0){
			reg_write(dev_lite,CSR_INTT_DONE,0);
 			break; 
		}
	} while(1);

	// read data
        dma_from_device(dev_dma_rd, INTT_DDR_ADDR, 524288, 0, 1, a); // 65536 * 8 byte = 524288
}


void MODB_FPGA(uint64_t *rxi, uint64_t *rai, uint64_t *rbi, uint64_t pr, uint64_t pri,uint64_t kbar2){
	uint32_t check_done=1;

   if (debug1) {printf("## Debug: function(%s()) line(%d)\n", __FUNCTION__, __LINE__); fflush(stdout);}
	//=== Write parameter
	uint32_t wpl = (uint32_t) (MASK_32_LOW & pr);          
	uint32_t wph = (uint32_t) ((pr & MASK_32_HIGH) >> 32);
        reg_write(dev_lite, CSR_MODB_P_L, wpl);        
        reg_write(dev_lite, CSR_MODB_P_H, wph);       

	wpl = (uint32_t) (MASK_32_LOW & pri);	
	wph = (uint32_t) ((pri & MASK_32_HIGH) >> 32);
        reg_write(dev_lite, CSR_MODB_PR_L, wpl);     
        reg_write(dev_lite, CSR_MODB_PR_H, wph);    
        reg_write(dev_lite, CSR_MODB_DDR, MODB_DDR_ADDR); 

	//=== Write parameter
	wpl = (uint32_t) (MASK_32_LOW & kbar2);          
	wph = (uint32_t) ((kbar2 & MASK_32_HIGH) >> 32); 
        reg_write(dev_lite, CSR_MODB_KBAR2_L, wpl);     
        reg_write(dev_lite, CSR_MODB_KBAR2_H, wph);    

	//== Write a
        dma_to_device(dev_dma_wt, MODB_A_ADDR, 524288, 0, 1, rai); // 65536 * 8 byte = 524288
 	//== Write b
        dma_to_device(dev_dma_wt, MODB_B_ADDR, 524288, 0, 1, rbi); // 65536 * 8 byte = 524288

	// check INTT Done
	do {
		check_done = reg_read(dev_lite, CSR_MODB_DONE);
		if(check_done != 0) {
			reg_write(dev_lite,CSR_MODB_DONE,0);
			break;
		}	
	} while(1);

	// read data
        dma_from_device(dev_dma_rd, MODB_DDR_ADDR, 524288, 0, 1, rxi); // 65536 * 8 byte = 524288
}
void MODB_FPGA_DMA(uint64_t *rxi, uint64_t *rai, uint64_t *rbi, uint32_t a_addr, uint32_t b_addr,uint32_t out_addr,uint64_t pr, uint64_t pri,uint64_t kbar2)
{
	char dev_lite[20]   = "/dev/xdma0_user";
	char dev_dma_wt[20] = "/dev/xdma0_h2c_0";
	char dev_dma_rd[20] = "/dev/xdma0_c2h_0";
	//variable
	uint32_t check_done=1;
	//=== Write parameter
	uint32_t wpl = (uint32_t) (MASK_32_LOW & pr);
	uint32_t wph = (uint32_t) ((pr & MASK_32_HIGH) >> 32);
    reg_write(dev_lite, CSR_MODB_P_L, wpl);
    reg_write(dev_lite, CSR_MODB_P_H, wph);
	wpl = (uint32_t) (MASK_32_LOW & pri);
	wph = (uint32_t) ((pri & MASK_32_HIGH) >> 32);
    reg_write(dev_lite, CSR_MODB_PR_L, wpl);
    reg_write(dev_lite, CSR_MODB_PR_H, wph);
    //fpga ddr address
    reg_write(dev_lite, CSR_MODB_DDR, out_addr);

	//=== Write parameter
	wpl = (uint32_t) (MASK_32_LOW & (kbar2-64));
	wph = (uint32_t) (((kbar2-64) & MASK_32_HIGH) >> 32);
    reg_write(dev_lite, CSR_MODB_KBAR2_L, wpl);
    reg_write(dev_lite, CSR_MODB_KBAR2_H, wph);
	//== Write a
    dma_to_device(dev_dma_wt, a_addr, 524288, 0, 1, rai); // 65536 * 8 byte = 524288
 	//== Write b
    dma_to_device(dev_dma_wt, b_addr, 524288, 0, 1, rbi); // 65536 * 8 byte = 524288
    //DMA transfer
    dma_transfer(a_addr,MODB_A_ADDR, 524288);
    dma_transfer(b_addr,MODB_B_ADDR, 524288);
	// check INTT Done
	do {
		check_done = reg_read(dev_lite, CSR_MODB_DONE);
		if(check_done != 0) {
			reg_write(dev_lite,CSR_MODB_DONE,0);
			break;
		}
	} while(1);
	// read data
    dma_from_device(dev_dma_rd, out_addr, 524288, 0, 1, rxi); // 65536 * 8 byte = 524288
}

double estimate_time(struct timespec *begin, struct timespec *end){
	double elapsed=(end->tv_sec - begin->tv_sec) + (end->tv_nsec - begin->tv_nsec)*1e-9;
	return elapsed;
}
