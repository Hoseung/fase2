
#include "cu_mult_handle.h"
#include "cu_crt.h"
#include "cu_icrt.h"
#include <cooperative_groups.h>
#include <cuda.h> 
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <chrono>
__constant__  long c_logN[1];

#define CUDA_API_PER_THREAD_DEFAULT_STEAM
#ifndef MAX_COEFF_SIZE
    #define MAX_COEFF_SIZE 64
#endif

std::chrono::high_resolution_clock::time_point n1, n2;
#define START() n1 = std::chrono::high_resolution_clock::now();
#define END() n2 = std::chrono::high_resolution_clock::now();
#define SUM(t) t += (double)std::chrono::duration_cast<std::hrono::nanoseconds >(n2-n1).count() ;
#define PRINTTIME(msg) std::cout << msg << " time = " << (double)std::chrono::duration_cast<std::chrono::nanoseconds >(n2 - n1).count() / 1000 << " microseconds" << std::endl;





__global__ void mulModBarrett_kernel(uint64_t* r,uint64_t* a,uint64_t* b, uint64_t* pVec, uint64_t* prVec, long* pTwok)
{
    uint64_t N = 1<<c_logN[0];
    int idx = blockIdx.y *blockDim.x + threadIdx.x;
	//unsigned __int128 mul = static_cast<unsigned __int128>(a) * b;
    uint64_t p = pVec[blockIdx.x];
    uint64_t pr = prVec[blockIdx.x];
    uint64_t twok = pTwok[blockIdx.x];
	uint64_t abot = a[blockIdx.x*N +idx ] * b[blockIdx.x*N +idx ];
	uint64_t atop = __umul64hi(a[blockIdx.x*N +idx ],b[blockIdx.x*N +idx ]);
    uint64_t tmp_high = 0;
    uint64_t tmp_low = __umul64hi(abot,pr);
    
    uint64_t tmp_atop_pr =  atop * pr;
    tmp_low += tmp_atop_pr;
    tmp_high = __umul64hi(atop,pr) + (tmp_low <tmp_atop_pr);
    tmp_low =  (tmp_low >> (twok - 64)) | (tmp_high << (128 -twok)) ;
    tmp_high >>= twok - 64;
	tmp_low =  tmp_low * p;
    tmp_high = tmp_high * p + __umul64hi(tmp_low,p);

    tmp_low = abot - tmp_low;
    r[blockIdx.x*N +idx] = tmp_low;
	if(r[blockIdx.x*N +idx ] >= p) {
		r[blockIdx.x*N +idx] -= p;
	}
}

__device__ void cuda_NTT_inner(uint64_t* a,long t, uint64_t W, uint64_t p,uint64_t pInv)
{
    uint64_t T = a[t];
    uint64_t U0 =  T* W;
    uint64_t U1 =  __umul64hi(T, W);
    uint64_t Q = U0 * pInv;
    uint64_t H =  __umul64hi(Q, p);
    uint64_t V = U1 < H ? U1 + p - H : U1 - H;
    a[t] = a[0] < V ? a[0] + p - V : a[0] - V;
    a[0] += V;
    if (a[0] > p)
        a[0] -= p;

}

__global__ void cuda_NTT_kernel(uint64_t* r, uint64_t* scaledRootInvPows, uint64_t* pVec, uint64_t* pInvVec) {
    cooperative_groups::grid_group grp = cooperative_groups::this_grid();
    long N = 1 << c_logN[0];
    long t = N;
     
    //printf("%ld\n",blockIdx.x << c_logN[0]);
    int idx = blockIdx.y *blockDim.x + threadIdx.x;
    for(long m=1;m<N;m<<=1)
    {
        t >>=1;
        int j = idx + t*(idx/t);
        uint64_t W = scaledRootInvPows[m + (idx/t)];

        cuda_NTT_inner(r+j,t,W,pVec[0],pInvVec[0]);
        grp.sync();
    }


}

__device__ void cuda_INTT_inner(uint64_t* a,long t, uint64_t W, uint64_t p,uint64_t pInv)
{
    uint64_t U = a[0] + a[t];
    if (U > p)
        U -= p;
    uint64_t T =a[0] < a[t] ? a[0] + p - a[t] : a[0] - a[t];
    uint64_t U0 =  T* W;
    uint64_t U1 =  __umul64hi(T, W);
    uint64_t Q = U0 * pInv;
    uint64_t H =  __umul64hi(Q, p);
    a[0] = U;
    a[t] = (U1 < H) ? U1 + p - H : U1 - H;


}

__device__ void cuda_INTT_scaler(uint64_t* a,long t, uint64_t NScale, uint64_t p,uint64_t pInv)
{
    uint64_t T = a[0];
    uint64_t U0 = T * NScale;
    uint64_t U1 =  __umul64hi(T, NScale);
    uint64_t Q = U0 * pInv;
    uint64_t H =__umul64hi(Q, p);
    a[0] = (U1 < H) ? U1 + p - H : U1 - H;

     T = a[t];
     U0 = T * NScale;
     U1 =  __umul64hi(T, NScale);
     Q = U0 * pInv;
     H = __umul64hi(Q, p);
    a[t] = (U1 < H) ? U1 + p - H : U1 - H;
}

__global__ void cuda_INTT_kernel(uint64_t* r, uint64_t* scaledRootPows, uint64_t* scaledNInv , uint64_t* pVec, uint64_t* pInvVec) 
{
    cooperative_groups::grid_group grp = cooperative_groups::this_grid();
    long N = 1 << c_logN[0];
    long t = 1;
     
    //printf("%ld\n",blockIdx.x << c_logN[0]);
    int idx = blockIdx.y *blockDim.x + threadIdx.x;

	for (long m=N; m > 1; m >>= 1) {
        long h = m >> 1;
        int j = idx + t*(idx/t);
        uint64_t W = scaledRootPows[ h + (idx/(t))];

        cuda_INTT_inner(r+j,t,W,pVec[0],pInvVec[0]);
        t <<= 1;
        
        grp.sync();
	}

    cuda_INTT_scaler(r+ idx,N>>1,scaledNInv[0],pVec[0],pInvVec[0]);

}

__global__ void test_print(uint64_t& test)
{
    printf("test: %lu  \n",test);
}

cuda_mult_handler::cuda_mult_handler(long nprimes_,long logN_)
{
    nprimes = nprimes_;
    logN = logN_;
    N = 1 << logN;
    coeff_size = nprimes *sizeof(uint64_t);

    crt_coeff_size  = coeff_size * N;
    cudaStreamCreate(&stream_mem);
    

    rem_builder = new cu_remainder_build(N,nprimes);
    icrt_builder = new cuda_reconstruct(N,nprimes);

}

cuda_mult_handler::~cuda_mult_handler()
{
    cudaFree(c_logN);
    // param for NTT
    cudaFree( c_pVec);
    cudaFree( c_prVec);
    cudaFree( c_pTwok);
    cudaFree( c_pInvVec);
    cudaFree( c_scaledRootPows);
    cudaFree( c_scaledRootInvPows);
    cudaFree( c_scaledNInv);
    cudaFree( c_logN);


    cudaStreamDestroy(stream_mem);

}

void cuda_mult_handler::ntt_memAlloc(long npimes_,long logN_)
{

    cudaMalloc(&c_pVec ,coeff_size);
    cudaMalloc(&c_prVec,coeff_size);
    cudaMalloc(&c_pInvVec,coeff_size);
    cudaMalloc(&c_pTwok,nprimes *sizeof(long));
    cudaMalloc(&c_scaledRootPows, coeff_size * N);
    cudaMalloc(&c_scaledRootInvPows, coeff_size * N);
    cudaMalloc(&c_scaledNInv, coeff_size);

    //(*icrt_builder).memAlloc();

    (*icrt_builder).memAlloc(nprimes);

}

void cuda_mult_handler::param_cpy(uint64_t* pVec, uint64_t* prVec, uint64_t* pInvVec, long* pTwok, uint64_t** scaledRootPows,uint64_t** scaledRootInvPows, uint64_t* scaledNInv, NTL::ZZ** pHat, uint64_t** pHatInvModp, unsigned long** coeffpinv_array, NTL::ZZ* pProd,NTL::ZZ* pProdh)
{
    
    cudaMemcpyAsync(c_pVec,pVec,coeff_size,cudaMemcpyHostToDevice,stream_mem);
    cudaMemcpyAsync(c_prVec,prVec,coeff_size,cudaMemcpyHostToDevice,stream_mem);
    cudaMemcpyAsync(c_pInvVec,pInvVec,coeff_size,cudaMemcpyHostToDevice,stream_mem);
    cudaMemcpyAsync(c_pTwok,pTwok,nprimes *sizeof(long),cudaMemcpyHostToDevice,stream_mem);
    for(long i=0; i <nprimes;i ++)
    {
        cudaMemcpyAsync(c_scaledRootPows + N*i,scaledRootPows[i], N*sizeof(uint64_t),cudaMemcpyHostToDevice,stream_mem);
        cudaMemcpyAsync(c_scaledRootInvPows + N*i,scaledRootInvPows[i], N*sizeof(uint64_t),cudaMemcpyHostToDevice,stream_mem);
    }

    cudaMemcpyAsync(c_scaledNInv,scaledNInv,coeff_size,cudaMemcpyHostToDevice,stream_mem);
    cudaMemcpyToSymbolAsync(c_logN,&logN,sizeof(long),0,cudaMemcpyHostToDevice,stream_mem);
    (*icrt_builder).c_pVec = c_pVec;

    (*icrt_builder).make_param(nprimes, pHat, pHatInvModp,coeffpinv_array, pProd, pProdh);
}

void cuda_mult_handler::param_position_set()
{
    (*icrt_builder).np = np;
    (*icrt_builder).c_pHatnp = (*icrt_builder).c_pHat + ((np-1)*np/2)* 64;
    (*icrt_builder).c_pHatInvModpnp = (*icrt_builder).c_pHatInvModp + (np-1)*np/2;
    (*icrt_builder).c_coeffpinv_arraynp = (*icrt_builder).c_coeffpinv_array + (np-1)*np/2;
    (*icrt_builder).c_pProdnp = (*icrt_builder).c_pProd + np-1;
    (*icrt_builder).c_pProdhnp = (*icrt_builder).c_pProdh + np-1;
}

uint64_t* cuda_mult_handler::host_uint_to_cuda_crt(uint64_signed x, cudaStream_t stream)
{

    if( !(*rem_builder).is_table_maked )
    {
        c_tbl = (*rem_builder).cuda_tbl_build(c_pVec,stream);
        (*rem_builder).is_table_maked = true;
    }
        



    uint64_t* c_data;// = mem_split_flag ? c_data_b : c_data_a;
    bool* c_neg_check;// = mem_split_flag ? c_neg_check_b : c_neg_check_a;

    cudaMalloc(&c_data,N*MAX_COEFF_SIZE*sizeof(uint64_t));
    cudaMalloc(&c_neg_check,N*sizeof(bool));
    //mem_split_flag = mem_split_flag ? false:true;
    cudaMemcpyAsync( c_data, x.data,N* MAX_COEFF_SIZE * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);  
    cudaMemcpyAsync( c_neg_check, x.neg_check,N*sizeof(bool),cudaMemcpyHostToDevice,stream);

    

    uint64_t *c_result = (*rem_builder).host_uint_to_cuda_remainder(c_data, c_neg_check, c_pVec, c_tbl,stream);
    cudaFree(c_data);
    cudaFree(c_neg_check);
    return c_result;
}

void cuda_mult_handler::ntt_poly_cpy(uint64_t* ra,uint64_t* rb)
{
    //cudaMemcpyAsync(c_ra,ra,crt_coeff_size,cudaMemcpyHostToDevice,stream_mem);
    //cudaMemcpyAsync(c_rb,rb,crt_coeff_size,cudaMemcpyHostToDevice,stream_mem);

    //c_ra = ra;
    //c_rb= rb;

}

uint64_t* cuda_mult_handler::ntt_to_host(uint64_t* c_x)
{

    uint64_t* x = new uint64_t[np << logN]();
    cudaMemcpyAsync(x, c_x,(np << logN)*sizeof(uint64_t),cudaMemcpyDeviceToHost,stream_mem);
    

    return x;
    //ntt_memFree();
}

void cuda_mult_handler::cuda_NTT_run(uint64_t* c_poly_ring,cudaStream_t stream, int thread_max )
{
    //cudaDeviceSynchronize(); 
    int block_max =0;
    if (logN-1 < thread_max)
    {
        thread_max = 1<<(logN-1);
        block_max = 1;
    }
    else
    {
        block_max = 1 << (logN-thread_max-1);
        thread_max =1 << thread_max;
    }
    dim3 grids(1,block_max,1);
    dim3 threads(thread_max,1,1);

    //multi_stream process
    long* m; cudaMalloc(&m, sizeof(long));
    cudaMemsetAsync(m,0,sizeof(long),stream);
    for(long i =0; i < np; i ++)
    {
        uint64_t* c_poly_ring_split = c_poly_ring + N*i;
        uint64_t* c_scaledRootPows_split = c_scaledRootPows + N*i;
        uint64_t* c_p = c_pVec + i;
        uint64_t* c_pInv = c_pInvVec +i;
        void* arguments[] = {(void*)&(c_poly_ring_split),  (void*)&c_scaledRootPows_split,(void*)& c_p, (void*)&c_pInv};
        
        
        //cout<< "NTT " <<cudaGetErrorString(
            cudaLaunchCooperativeKernel((void*) cuda_NTT_kernel, grids, threads, arguments,0,stream)
        ;//) <<endl;
        //cudaStreamDestroy(stream);
    }        
}

void cuda_mult_handler::cuda_INTT_run(uint64_t* c_poly_ring,int thread_max )
{
    //cudaDeviceSynchronize(); 
    int block_max =0;
    if (logN-1 < thread_max)
    {
        thread_max = 1<<(logN-1);
        block_max = 1;
    }
    else
    {
        block_max = 1 << (logN-thread_max-1);
        thread_max =1 << thread_max;
    }
    dim3 grids(1,block_max,1);
    dim3 threads(thread_max,1,1);

    //multi_stream process
    for(long i =0; i < np; i ++)
    {
        // cudaStream_t stream;
        // cudaStreamCreate(&stream);
        uint64_t* c_poly_ring_split = c_poly_ring + N*i;
        uint64_t* c_scaledRootInvPows_split = c_scaledRootInvPows + N*i;
        uint64_t* c_p = c_pVec + i;
        uint64_t* c_pInv = c_pInvVec +i;
        uint64_t* c_scaled = c_scaledNInv +i;
        void* arguments[] = {(void*)&(c_poly_ring_split), (void*)&c_scaledRootInvPows_split, (void*)&c_scaled,(void*)& c_p, (void*)&c_pInv};
        
        //cout <<cudaGetErrorString(
            cudaLaunchCooperativeKernel((void*) cuda_INTT_kernel, grids, threads, arguments,0)
        ;//) <<endl;
        //cudaStreamDestroy(stream);
    }        
}

uint64_t* cuda_mult_handler::mulModBarrett_cuda(uint64_t* c_ra,uint64_t* c_rb ,int thread_max)
{

    uint64_t* c_rx;
    cudaMalloc(&c_rx,crt_coeff_size);

    int block_max =0;
    if (logN < thread_max)
    {
        thread_max = 1<<(logN);
        block_max = 1;
    }
    else
    {
        block_max = 1 << (logN-thread_max);
        thread_max =1 << thread_max;
    }
    dim3 grids(np,block_max,1);
    dim3 threads(thread_max,1,1);
    mulModBarrett_kernel<<<grids,threads>>>(c_rx,c_ra,c_rb,c_pVec,c_prVec,c_pTwok );

    return c_rx;
}

void cuda_mult_handler::cuda_uint_to_host_icrt(NTL::ZZ* x,uint64_t* c_rx ,const NTL::ZZ& mod)
{
    (*icrt_builder).cuda_icrt_run(x,c_rx,mod);
}

uint64_signed cuda_mult_handler::zz_to_uint(NTL::ZZ* data)
{
    uint64_t* data_tmp = new uint64_t[MAX_COEFF_SIZE * N]{0,};
    bool* neg_check = new bool[N];
    //
    for(long j =0;j<N;j++)
    {
      NTL::ZZ a_data = data[j];
      long size = a_data.size();
      if(size == 0 ) size = 1;
      
      if (a_data < 0)
      {
        a_data = -a_data;
        neg_check[j] = true;
      }
      else{
        neg_check[j] = false;
      }

      //uint64_t* tmp = new uint64_t[size]{0,};
      for(long i =0;i < size ; i++)
      {
        data_tmp[i + j*MAX_COEFF_SIZE] = NTL::conv<uint64_t>(a_data);
        a_data >>= 64;
      }
      //cudaMemcpy( c_data+j*MAX_COEFF_SIZE, tmp,(size)*sizeof(uint64_t),cudaMemcpyHostToDevice);
    }
    uint64_signed uint64_data;
    uint64_data.data = data_tmp;
    uint64_data.neg_check = neg_check;

    //delete[] data_tmp;
    //delete[] neg_check;
    return uint64_data;
}

uint64_t* cuda_mult_handler::host_ntt_to_cuda_ntt(uint64_t* ra,cudaStream_t stream_mem)
{
    uint64_t* result;
    cudaMalloc(&result,np*N*sizeof(uint64_t));
    cudaMemcpyAsync(result, ra, np*N*sizeof(uint64_t),cudaMemcpyHostToDevice,stream_mem);
    return result;
}