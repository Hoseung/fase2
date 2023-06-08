#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <NTL/ZZ.h>
#ifndef CGBN_USE
    #include <gmp.h>
    #include "cgbn/cgbn.h"
#define CGBN_USE
#endif

#define TPI 32
#define BITS 4096
#define INSTANCES 1

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

class cuda_reconstruct
{
private:

public:
    uint64_t* c_pHatnp;
    
    
    uint64_t* c_pVec;
    uint64_t* c_pHatInvModpnp;
    uint64_t* c_coeffpinv_arraynp;
    unsigned char * pp;
    cgbn_mem_t<BITS>* tmp_cgbn; 

    cgbn_mem_t<BITS>* c_pProdnp,* c_pProdhnp;

    uint64_t* c_pHat;
    uint64_t* c_pHatInvModp;
    uint64_t* c_coeffpinv_array;
    cgbn_mem_t<BITS>* c_pProd;
    cgbn_mem_t<BITS>* c_pProdh;

    

    long np,n;
    cuda_reconstruct(long _n,long _np);
    ~cuda_reconstruct();
    void memAlloc(long nprime);
    void copy_param(long count,NTL::ZZ* pHatnp, uint64_t* pHatInvModpnp, unsigned long* coeffpinv_arraynp, NTL::ZZ& pProdnp,NTL::ZZ& pProdhnp);
    
    void reconst(uint64_t* c_result,uint64_t* rx);
    void cuda_icrt_run(NTL::ZZ* x,uint64_t* c_rx ,const NTL::ZZ& mod);
    cgbn_mem_t<BITS> * uint_to_cgbn(const uint64_t* c_input,long size,int thread_level=9);
    uint64_t* cgbn_to_uint(cgbn_mem_t<BITS>* c_input,long size,int thread_level=9);
    cgbn_mem_t<BITS> * cgbn_multi_div_one(cgbn_mem_t<BITS> * a,cgbn_mem_t<BITS> * b,long size,int thread_level =9);
    void cu_return_data(NTL::ZZ* x,uint64_t* c_result,const NTL::ZZ& mod);
    void param_zz_to_cuda(cgbn_mem_t<BITS>* c_result,NTL::ZZ& input_x);
    void param_zz_to_cuda(cgbn_mem_t<BITS>* c_result, const NTL::ZZ& input_x);

    void memFree();
    void make_param(long nprime, NTL::ZZ** pHat, uint64_t** pHatInvModp, unsigned long** coeffpinv_array, NTL::ZZ* pProd,NTL::ZZ* pProdh);
};
