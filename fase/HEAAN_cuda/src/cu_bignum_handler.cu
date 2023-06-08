#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gmp.h>
//#include "cgbn/cgbn.h"
#include "cgbn.h"
#include "cu_bignum_handler.h"

#define TPI 32
#define BITS 64*64
#define MAX_SIZE 64

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS> cgbn_mem;


__global__ void uint_to_cgbn(cgbn_mem* output, uint64_t* input)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
      /*
      for(long i =0;i<MAX_SIZE;i++)
      {
        output[idx]._limbs[2*i] = (uint32_t) input[MAX_SIZE * idx +i] ;
        output[idx]._limbs[2*i+1]   = (uint32_t)(input[MAX_SIZE * idx +i] >> 32);
      }
      printf("\n\n");
      */
    for(int i =0; i <64;i++)
    {
      printf("%x  %x  ",output[idx]._limbs[2*i],output[idx]._limbs[2*i+1]);
    }
}

__global__ void cgbn_to_uint(uint64_t* output, cgbn_mem*  input)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    /*
    printf("\n\n");
    for(int i =0; i <64;i++)
    {
      printf("%x  %x  ",input[idx]._limbs[2*i],input[idx]._limbs[2*i+1]);
    }
    printf("\n\n");
    */


    for(long i =0;i<MAX_SIZE;i++)
    {
      output[MAX_SIZE * idx +i] =  (uint64_t) input[idx]._limbs[2*i] + ((uint64_t) input[idx]._limbs[2*i+1] << 32);
    }
}

__global__ void kernel_div(cgbn_error_report_t *report, cgbn_mem * gx, cgbn_mem * ga, cgbn_mem * gb) 
{
  // decode an instance number from the blockIdx and threadIdx
    int32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;

    context_t      bn_context(cgbn_report_monitor, report, instance);   
    env_t          bn_env(bn_context.env<env_t>());                     
    env_t::cgbn_t  a, b, r;                                             


    cgbn_load(bn_env, a, &ga[instance]);      
    cgbn_load(bn_env, b, &gb[instance]);      

    cgbn_add(bn_env, r, a, b);                           
    
    cgbn_store(bn_env, &gx[instance], r);  
}



cu_bignum_handler::cu_bignum_handler(long _N,long _np)
{
    N =_N;
    np = np;
 

}
cu_bignum_handler::~cu_bignum_handler()
{
}
void cu_bignum_handler::uint_to_cgbn(uint64_t* x)
{

}

void cu_bignum_handler::test()
{
    for(int i =0;i<30;i++)
    {
        printf("hello World\n");
    }
}