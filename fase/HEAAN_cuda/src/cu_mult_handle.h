#include <NTL/ZZ.h>

typedef struct uint64_signed {
    uint64_t* data;
    bool* neg_check;
} uint64_signed;


class cu_remainder_build;
class cuda_reconstruct;

class cuda_mult_handler
{
private:
public:
    
    long coeff_size;
    long crt_coeff_size;
    long nprimes;
    long np;
    long logN;
    long N;

    // param for NTT
    uint64_t* c_pVec;
    uint64_t* c_prVec;
    uint64_t* c_pInvVec;
    long* c_pTwok;
    uint64_t* c_scaledRootPows;
    uint64_t* c_scaledRootInvPows;
    uint64_t* c_scaledNInv;

    // poly for NTT
    //uint64_t* c_ra;
    //uint64_t* c_rb;
    //uint64_t* c_rx;
    cudaStream_t stream_mem;

    cu_remainder_build* rem_builder;
    cuda_reconstruct* icrt_builder;

    uint64_t* c_tbl;



cuda_mult_handler(long np_,long logN_);

~cuda_mult_handler();

void ntt_memAlloc(long np_,long logN_);

void param_cpy(uint64_t* pVec, uint64_t* prVec, uint64_t* pInvVec, long* pTwok, uint64_t** scaledRootPows,uint64_t** scaledRootInvPows, uint64_t* scaledNInv, NTL::ZZ** pHat, uint64_t** pHatInvModp, unsigned long** coeffpinv_array, NTL::ZZ* pProd,NTL::ZZ* pProdh);

void param_position_set();

uint64_t* host_uint_to_cuda_crt(uint64_signed x,cudaStream_t stream);

void ntt_poly_cpy(uint64_t* ra,uint64_t* rb);

uint64_t* ntt_to_host(uint64_t* x);

void cuda_NTT();

void cuda_INTT();

void cuda_NTT_run(uint64_t* c_poly_ring, cudaStream_t stream,int thread_max = 8);

void cuda_INTT_run(uint64_t* c_poly_ring,int thread_max = 8);


uint64_t* mulModBarrett_cuda(uint64_t* c_ra,uint64_t* c_rb, int thread_max = 8);

void cuda_uint_to_host_icrt(NTL::ZZ* x,uint64_t* c_rx ,const NTL::ZZ& mod);

uint64_signed zz_to_uint(NTL::ZZ* x);

bool* zz_to_sign(NTL::ZZ* x);

uint64_t* host_ntt_to_cuda_ntt(uint64_t* ra,cudaStream_t stream_mem);

};