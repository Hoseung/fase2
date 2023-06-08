/*
 * rand_gen.cpp
 *
 *  Created on: Oct 22, 2020
 *      Author: skim
 */

#include <iostream>
#include <cmath>

//bit-width -> max_value
void rand_gen(uint64_t *rand_arr,int num, int nbit){
	uint32_t uRand,bRand;
	uint64_t rand_num;
	int num_mask=64-nbit;
	uint64_t mask_bit=0;

	if(num_mask>0){
		mask_bit=0;
		for(int j=0;j<num_mask;j++){
			mask_bit |= 1ULL<<(64-j-1);
		}
	}
	mask_bit = ~mask_bit;
	//printf("mask_bit=%016llx\n",mask_bit);
	for (int i=0;i<num;i++){
		uRand=(uint32_t)rand();
		bRand=(uint32_t)rand();
		//rand_arr[i]=((uint64_t)uRand <<32)+bRand;
		rand_num=((uint64_t)uRand <<32)+bRand;
		rand_arr[i]=rand_num & mask_bit;
	}
}
void rand_gen_top(){
	const int num=64;
	const int nbit=63;
	uint64_t rand_arr[num]={0,};
	rand_gen(rand_arr,num,nbit);
	for(int i=0;i<num;i++){
		printf("rand_arr[%d]:dec=%llu,hex=%016llx\n",i,rand_arr[i],rand_arr[i]);
	}

}
/*

void hex_print(){
	uint64_t rand_num;
	uint64_t max_val;
	max_val=1ULL<<63;
	//max_val= ~0ULL; //maximum value for unsigned 64-bit number
	//max_val=max_val<<30;
	printf("max_val=%016llx,RAND_MAX=%016llx\n",max_val,RAND_MAX);
	for(int i=0;i<10;i++){
		rand_num=getRandom(0,max_val);
		printf("i=%d:rand_num=%llu,rand_num_hex=%016llx\n",i,rand_num,rand_num);
	}
}
*/
