/*
 * rand_gen.h
 *
 *  Created on: Oct 22, 2020
 *      Author: skim
 */

#ifndef RAND_GEN_H_
#define RAND_GEN_H_

#include <stdint.h>
void rand_gen(uint64_t *rand_arr,int num, int nbit);
void rand_gen_top();
void hex_print();


#endif /* RAND_GEN_H_ */
