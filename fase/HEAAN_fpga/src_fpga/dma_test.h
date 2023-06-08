/*
 * dma_test.h
 *
 *  Created on: Mar 8, 2021
 *      Author: etri_ai3
 */

#ifndef DMA_TEST_H_
#define DMA_TEST_H_


void host2mem_rw_test();
void simple_dma_test();
void dma_reg_test();
void dma_transfer(uint32_t src_addr,uint32_t dst_addr, uint32_t num_bytes);


#endif /* DMA_TEST_H_ */
