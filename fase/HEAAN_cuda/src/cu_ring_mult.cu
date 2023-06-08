

#include "RingMultiplier.h"

#include "cu_mult_handle.h"
#include <chrono>
#include <cuda.h>


chrono::high_resolution_clock::time_point g1, g2;
#define START() g1 = chrono::high_resolution_clock::now();
#define END() g2 = chrono::high_resolution_clock::now();
#define SUM(t) t += (double)chrono::duration_cast<chrono::nanoseconds >(g2-g1).count() ;
#define PRINTTIME(msg) std::cout << msg << " time = " << (double)chrono::duration_cast<std::chrono::nanoseconds >(g2 - g1).count() / 1000 << " microseconds" << std::endl;


void RingMultiplier::cuda_instanciate(long nprimes, long logN)
{
    
    mult_handle = new cuda_mult_handler(nprimes,logN);

    (*mult_handle).ntt_memAlloc(nprimes,logN);
    
    (*mult_handle).param_cpy(pVec,prVec,pInvVec,pTwok,scaledRootPows,scaledRootInvPows,scaledNInv,
                             pHat,pHatInvModp,coeffpinv_array,pProd,pProdh);


}

void RingMultiplier::cuda_host_mult(NTL::ZZ* x, NTL::ZZ* a, NTL::ZZ* b, long np, const NTL::ZZ& mod)
{
    
    //chrono::high_resolution_clock::time_point start, end;
    //start = chrono::high_resolution_clock::now();
    (*mult_handle).np = np;


    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);
    
    //START()
    uint64_signed ra = (*mult_handle).zz_to_uint(a);
    uint64_signed rb = (*mult_handle).zz_to_uint(b);
    //END()PRINTTIME("GPU: ZZ_to_INT : ")

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;
    //START()
    t_a = (*mult_handle).host_uint_to_cuda_crt(ra,stream_a);
    t_b = (*mult_handle).host_uint_to_cuda_crt(rb,stream_b);
    //END()PRINTTIME("GPU: CRT : ")

    //START()
    (*mult_handle).cuda_NTT_run(t_a,stream_a);
    (*mult_handle).cuda_NTT_run(t_b,stream_b);
    //END()PRINTTIME("GPU: NTT : ")

    //START()
    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);
    //END()PRINTTIME("GPU: MULT : ")

    //START()
    (*mult_handle).cuda_INTT_run(t_x);
    //END()PRINTTIME("GPU: INTT : ")

    (*mult_handle).param_position_set();

    //START()
    (*mult_handle).cuda_uint_to_host_icrt(x,t_x,mod);
    //END()PRINTTIME("GPU: ICRT : ")


    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);
    delete[] ra.data;
    delete[] ra.neg_check;
    delete[] rb.data;
    delete[] rb.neg_check;
    //end = chrono::high_resolution_clock::now();
    //cout<<" GPU: FULL TIME :" <<  (double)chrono::duration_cast<std::chrono::nanoseconds >(end - start).count() / 1000<<  endl;
}

void RingMultiplier::cuda_multNTT(NTL::ZZ* x, NTL::ZZ* a, uint64_t* rb, long np,const NTL::ZZ& mod)
{
    (*mult_handle).np = np;

    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);
    
    uint64_signed ra = (*mult_handle).zz_to_uint(a);

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;

    t_a = (*mult_handle).host_uint_to_cuda_crt(ra,stream_a);

    t_b = (*mult_handle).host_ntt_to_cuda_ntt(rb,stream_b);

    (*mult_handle).cuda_NTT_run(t_a,stream_a);

    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);
    
    (*mult_handle).cuda_INTT_run(t_x);
    
    (*mult_handle).param_position_set();
    
    (*mult_handle).cuda_uint_to_host_icrt(x,t_x,mod);

    //printf("called multNTT\n");
    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);
    delete[] ra.data;
    delete[] ra.neg_check;
}

void RingMultiplier::cuda_multDNTT(NTL::ZZ* x, uint64_t* ra, uint64_t* rb, long np,const NTL::ZZ& mod)
{
    (*mult_handle).np = np;

    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;

    t_a = (*mult_handle).host_ntt_to_cuda_ntt(ra,stream_a);

    t_b = (*mult_handle).host_ntt_to_cuda_ntt(rb,stream_b);
    
    cudaDeviceSynchronize();
    
    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);
    
    (*mult_handle).cuda_INTT_run(t_x);
    
    (*mult_handle).param_position_set();
    
    (*mult_handle).cuda_uint_to_host_icrt(x,t_x,mod);

    //printf("called multNTT\n");
    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);

}

void RingMultiplier::cuda_multAndEqual(NTL::ZZ* a, NTL::ZZ* b, long np,const NTL::ZZ& mod) 
{
    (*mult_handle).np = np;


    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);
    

    uint64_signed ra = (*mult_handle).zz_to_uint(a);
    uint64_signed rb = (*mult_handle).zz_to_uint(b);

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;
    t_a = (*mult_handle).host_uint_to_cuda_crt(ra,stream_a);
    t_b = (*mult_handle).host_uint_to_cuda_crt(rb,stream_b);

    (*mult_handle).cuda_NTT_run(t_a,stream_a);
    (*mult_handle).cuda_NTT_run(t_b,stream_b);

    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);

    (*mult_handle).cuda_INTT_run(t_x);

    (*mult_handle).param_position_set();

    (*mult_handle).cuda_uint_to_host_icrt(a,t_x,mod);


    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);
    delete[] ra.data;
    delete[] ra.neg_check;
    delete[] rb.data;
    delete[] rb.neg_check;

}

void RingMultiplier::cuda_multNTTAndEqual(NTL::ZZ* a, uint64_t* rb, long np,const NTL::ZZ& mod)
{
    (*mult_handle).np = np;

    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);
    
    uint64_signed ra = (*mult_handle).zz_to_uint(a);

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;

    t_a = (*mult_handle).host_uint_to_cuda_crt(ra,stream_a);

    t_b = (*mult_handle).host_ntt_to_cuda_ntt(rb,stream_b);

    (*mult_handle).cuda_NTT_run(t_a,stream_a);

    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);
    
    (*mult_handle).cuda_INTT_run(t_x);
    
    (*mult_handle).param_position_set();
    
    (*mult_handle).cuda_uint_to_host_icrt(a,t_x,mod);

    //printf("called multNTT\n");
    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);
    delete[] ra.data;
    delete[] ra.neg_check;
}

void RingMultiplier::cuda_square(NTL::ZZ* x, NTL::ZZ* a, long np,const NTL::ZZ& mod) 
{
    (*mult_handle).np = np;


    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);
    

    uint64_signed ra = (*mult_handle).zz_to_uint(a);
    uint64_signed rb = (*mult_handle).zz_to_uint(a);

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;
    t_a = (*mult_handle).host_uint_to_cuda_crt(ra,stream_a);
    t_b = (*mult_handle).host_uint_to_cuda_crt(rb,stream_b);

    (*mult_handle).cuda_NTT_run(t_a,stream_a);
    (*mult_handle).cuda_NTT_run(t_b,stream_b);

    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);

    (*mult_handle).cuda_INTT_run(t_x);

    (*mult_handle).param_position_set();

    (*mult_handle).cuda_uint_to_host_icrt(x,t_x,mod);


    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);
    delete[] ra.data;
    delete[] ra.neg_check;
    delete[] rb.data;
    delete[] rb.neg_check;
}

void RingMultiplier::cuda_squareNTT(NTL::ZZ* x, uint64_t* ra,long np,const NTL::ZZ& mod)
{
    (*mult_handle).np = np;

    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;

    t_a = (*mult_handle).host_ntt_to_cuda_ntt(ra,stream_a);

    t_b = (*mult_handle).host_ntt_to_cuda_ntt(ra,stream_b);
    
    cudaDeviceSynchronize();
    
    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);
    
    (*mult_handle).cuda_INTT_run(t_x);
    
    (*mult_handle).param_position_set();
    
    (*mult_handle).cuda_uint_to_host_icrt(x,t_x,mod);

    //printf("called multNTT\n");
    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);
}

void RingMultiplier::cuda_squareAndEqual(NTL::ZZ* a, long np,const NTL::ZZ& mod) 
{
    (*mult_handle).np = np;


    cudaStream_t stream_a; cudaStreamCreate(&stream_a);
    cudaStream_t stream_b; cudaStreamCreate(&stream_b);
    

    uint64_signed ra = (*mult_handle).zz_to_uint(a);
    uint64_signed rb = (*mult_handle).zz_to_uint(a);

    cudaDeviceSynchronize();
    

    uint64_t* t_a,*t_b,*t_x;
    t_a = (*mult_handle).host_uint_to_cuda_crt(ra,stream_a);
    t_b = (*mult_handle).host_uint_to_cuda_crt(rb,stream_b);

    (*mult_handle).cuda_NTT_run(t_a,stream_a);
    (*mult_handle).cuda_NTT_run(t_b,stream_b);

    t_x = (*mult_handle).mulModBarrett_cuda(t_a,t_b);

    (*mult_handle).cuda_INTT_run(t_x);

    (*mult_handle).param_position_set();

    (*mult_handle).cuda_uint_to_host_icrt(a,t_x,mod);


    cudaFree(t_a);
    cudaFree(t_b);
    cudaFree(t_x);
    delete[] ra.data;
    delete[] ra.neg_check;
    delete[] rb.data;
    delete[] rb.neg_check;
}