#include "cu_icrt.h"
#include <chrono>
std::chrono::high_resolution_clock::time_point gic1, gic2;
#define START() gic1 = std::chrono::high_resolution_clock::now();
#define END() gic2 = std::chrono::high_resolution_clock::now();
#define PRINTTIME(msg) std::cout << msg << " time = " << (double)std::chrono::duration_cast<std::chrono::nanoseconds >(gic2 - gic1).count() / 1000 << " microseconds" << std::endl;

#define MAX_SIZE 64
#define NP_MAX 64

__constant__  long c_n[1];
__constant__  long c_np[1];


__device__ void add_big_type( uint64_t* a,uint64_t* b)
{
    uint64_t carry = 0;
    for(int i =0;i<MAX_SIZE;i++)
    {
        a[i] =  a[i] + b[i] + carry;
        carry = (a[i] < b[i]) ? 1 : 0;

    }
}

__device__ void mult_big_type(uint64_t* result, uint64_t* a,long s)
{
    uint64_t carry = 0;
    uint64_t tmp =0;
    for(int i =0;i<MAX_SIZE;i++)
    {
        tmp = a[i]*s;
        tmp += carry;

        carry = __umul64hi(a[i],s) + (tmp < carry);

        result[i] = tmp;
    }
    
}

__device__ long reconst_inner(uint64_t rxi,uint64_t tt,uint64_t p,unsigned long ttpinv )
{
    uint64_t qq = __umul64hi(rxi,ttpinv);
    uint64_t rr = rxi * tt - qq * p;
    long rrl = long(rr);
    long pl = long(p);
    return rrl -pl >=0? rrl-pl : rrl;
}

__global__ void cu_reconst(uint64_t* result,uint64_t* pHatnp, uint64_t* rx,uint64_t* pVec,uint64_t* pHatInvModpnp,uint64_t* coeffpinv_arraynp)
{
    __shared__ uint64_t acc[64*MAX_SIZE];
    if(threadIdx.x ==0)
    {
      for(int i =0;i<64*MAX_SIZE;i++)
      {
          acc[i] =0;
          //if(blockIdx.x ==0) printf("result %d %lu\n",i, acc[i]);
      }
    }
    __syncthreads();

    uint64_t rxi = rx[blockIdx.x + (threadIdx.x *c_n[0])];
    long s = reconst_inner(rxi,pHatInvModpnp[threadIdx.x],pVec[threadIdx.x],coeffpinv_arraynp[threadIdx.x]);
 
    // if(blockIdx.x ==0 && threadIdx.x ==0 ){for(int i =0;i<64;i++){printf(" %lu ",acc[i]);} printf("\n\n");}
    // if(blockIdx.x ==0 && threadIdx.x ==0 ){for(int i =0;i<64;i++){printf(" %lu ",pHatnp[i]);} printf("\n\n");}
    

    mult_big_type(acc + MAX_SIZE*threadIdx.x, pHatnp + MAX_SIZE*threadIdx.x,s);
    

    if(threadIdx.x ==0)
    {
      for(int i = blockDim.x; i<64; i++)
      {
          for(int j=0;j<64;j++)
          {
            acc[i*MAX_SIZE +j] =0;
          }
          //if(blockIdx.x ==0) printf("result %d %lu\n",i, acc[i]);
      }
    }
    __syncthreads();


    for(int i = NP_MAX>>1 ; i >0; i>>=1)
    {
      if(threadIdx.x < i && threadIdx.x+i < blockDim.x)
      {
        add_big_type(acc + MAX_SIZE*threadIdx.x,  acc + MAX_SIZE*(threadIdx.x+i));
      }
      __syncthreads();   
    }
    __syncthreads();

    if(threadIdx.x ==0)
    {
      for(int i =0;i<MAX_SIZE;i++)
      {
          result[i + MAX_SIZE*blockIdx.x] = acc[i];
          //if(blockIdx.x ==0) printf("result %d %lu\n",i, acc[i]);
      }
      __syncthreads();
    }

}

__global__ void cu_rem( cgbn_mem_t<BITS> *result, cgbn_mem_t<BITS> *num, cgbn_mem_t<BITS> *pProdnp, cgbn_mem_t<BITS> *pProdhnp, cgbn_mem_t<BITS> *mod) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;

  context_t      bn_context(cgbn_report_monitor);
  env_t          bn_env(bn_context.env<env_t>());
  env_t::cgbn_t  a,r, en_pProdnp,en_pProdhnp, en_mod;
    int32_t comp;
  cgbn_load(bn_env, a, &num[instance]);
  cgbn_load(bn_env, en_pProdnp, &pProdnp[0]);
  cgbn_load(bn_env, en_pProdhnp, &pProdhnp[0]);
  cgbn_load(bn_env, en_mod, &mod[0]);
  cgbn_rem(bn_env, r, a, en_pProdnp);
  comp = cgbn_compare(bn_env, r, en_pProdhnp);
  if (comp == 1)
  {
      cgbn_sub(bn_env,r, en_pProdnp, r);
  } 
    cgbn_rem(bn_env, r, r, en_mod);
    if (comp == 1)
  {
      cgbn_sub(bn_env,r, en_mod, r);
  }  

  cgbn_store(bn_env, &(result[instance]), r);
}


__global__ void uint64_to_cgbn(cgbn_mem_t<BITS> * result,uint64_t* inputs )
{

    result[blockIdx.x]._limbs[2*threadIdx.x] = (uint32_t)inputs[blockIdx.x*MAX_SIZE +threadIdx.x];
    result[blockIdx.x]._limbs[2*threadIdx.x+1] = (uint32_t)(inputs[blockIdx.x*MAX_SIZE +threadIdx.x]>>32);


}


__global__ void cgbn_to_uint64(uint64_t * result, cgbn_mem_t<BITS>* inputs )
{
    long instance = blockIdx.x*blockDim.x + threadIdx.x;
    for(long i =0;i <MAX_SIZE;i++)
    {
        result[instance*MAX_SIZE +i] = 
        (uint64_t)inputs[instance]._limbs[2*i] |
        (uint64_t)inputs[instance]._limbs[2*i+1] <<32 ; 
        
    }
}

cuda_reconstruct::cuda_reconstruct(long _n,long _np)
{
    n = _n;
    np = _np;


}

cuda_reconstruct::~cuda_reconstruct()
{
    memFree();
}

void cuda_reconstruct::memFree()
{
    cudaFree(c_pHatnp);
    cudaFree(c_pHatInvModp);
    cudaFree(c_coeffpinv_array);

    cudaFree(c_pProd);
    cudaFree(c_pProdh);

    delete[] pp;
    delete[] tmp_cgbn;
}

void cuda_reconstruct::memAlloc(long nprime)
{
    
    pp =  new unsigned char[MAX_SIZE*8];
    tmp_cgbn = new cgbn_mem_t<BITS>[1];
    cudaMalloc(&c_pHat,(nprime*(nprime+1)/2) * MAX_SIZE * sizeof(uint64_t));
    cudaMalloc(&c_pHatInvModp,(nprime*(nprime+1)/2) * sizeof(uint64_t));
    cudaMalloc(&c_coeffpinv_array,(nprime*(nprime+1)/2) * sizeof(uint64_t));
    cudaMalloc(&c_pProd, nprime * sizeof(cgbn_mem_t<BITS>));
    cudaMalloc(&c_pProdh, nprime * sizeof(cgbn_mem_t<BITS>));

    

    //cudaMalloc((void **)&c_num, sizeof(cgbn_mem_t<BITS>)*n);
    //cudaMalloc((void **)&c_mod, sizeof(cgbn_mem_t<BITS>)*1);
    //cudaMalloc((void **)&c_r, sizeof(cgbn_mem_t<BITS>)*n);

    cudaMemcpyToSymbol(c_n,&n,sizeof(long),0,cudaMemcpyHostToDevice);
}


void cuda_reconstruct::copy_param(long count,NTL::ZZ* pHatnp, uint64_t* pHatInvModpnp, unsigned long* coeffpinv_arraynp, NTL::ZZ& pProdnp,NTL::ZZ& pProdhnp)
{

    uint64_t *tmp_pHatnp = new uint64_t[np*MAX_SIZE];
    for(int j = 0; j < count;j ++)
    {
    
        BytesFromZZ(pp, pHatnp[j], NTL::NumBytes(pHatnp[j]));
        for(int i =0;i  < MAX_SIZE * 8 ; i++ )
        {
            if(i%8==0)
            {
            tmp_pHatnp[MAX_SIZE*j+ i/8] = 0;  
            } 
            if(i < NTL::NumBytes(pHatnp[j]) )
            {
                tmp_pHatnp[MAX_SIZE*j+ i/8] +=  (uint64_t)pp[i] <<(8*(i%8));
            }
        }
    }

    cudaMemcpy(c_pHatnp, tmp_pHatnp, count* MAX_SIZE *sizeof(uint64_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(c_pVec,pVec,np*sizeof(uint64_t),cudaMemcpyHostToDevice);
    cudaMemcpy(c_pHatInvModpnp, pHatInvModpnp, count*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(c_coeffpinv_arraynp, coeffpinv_arraynp, count*sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    param_zz_to_cuda(c_pProdnp,pProdnp);
   
    param_zz_to_cuda(c_pProdhnp,pProdhnp);

    delete[] tmp_pHatnp;

}

void cuda_reconstruct::make_param(long nprime, NTL::ZZ** pHat, uint64_t** pHatInvModp, unsigned long** coeffpinv_array, NTL::ZZ* pProd,NTL::ZZ* pProdh)
{

     for(int i = 0; i < nprime; i ++)
     {
         c_pHatnp = c_pHat + ((i+1)*i/2)* MAX_SIZE;
         c_pHatInvModpnp = c_pHatInvModp + (i+1)*i/2;
         c_coeffpinv_arraynp = c_coeffpinv_array + (i+1)*i/2;
         c_pProdnp = c_pProd + i;
         c_pProdhnp = c_pProdh + i;

         copy_param(i+1,pHat[i], pHatInvModp[i],coeffpinv_array[i], pProd[i], pProdh[i]);
     }


}


void cuda_reconstruct::param_zz_to_cuda(cgbn_mem_t<BITS>* c_result,NTL::ZZ& input_x)
{
   
    BytesFromZZ(pp, input_x, NTL::NumBytes(input_x));
    for(int i =0;i  < MAX_SIZE * 8 ; i++ )
    {
        if(i%4==0)
        {
          tmp_cgbn[0]._limbs[i/4] = 0;  
        } 
        if(i < NTL::NumBytes(input_x) )
        {
            tmp_cgbn[0]._limbs[i/4] +=  (uint32_t)pp[i] <<(8*(i%4));
        }
    }

    cudaMemcpy(c_result, tmp_cgbn, sizeof(cgbn_mem_t<BITS>)*1, cudaMemcpyHostToDevice);
}


void cuda_reconstruct::param_zz_to_cuda(cgbn_mem_t<BITS>* c_result, const NTL::ZZ& input_x)
{
   
    BytesFromZZ(pp, input_x, NTL::NumBytes(input_x));
    for(int i =0;i  < MAX_SIZE * 8 ; i++ )
    {
        if(i%4==0)
        {
          tmp_cgbn[0]._limbs[i/4] = 0;  
        } 
        if(i < NTL::NumBytes(input_x) )
        {
            tmp_cgbn[0]._limbs[i/4] +=  (uint32_t)pp[i] <<(8*(i%4));
        }
    }

    cudaMemcpy(c_result, tmp_cgbn, sizeof(cgbn_mem_t<BITS>)*1, cudaMemcpyHostToDevice);
}



void cuda_reconstruct::cu_return_data(NTL::ZZ* x,uint64_t* c_result, const NTL::ZZ& mod)
{
    cgbn_mem_t<BITS> * c_num, * c_r, * c_mod;
    cudaMalloc((void **)&c_num, sizeof(cgbn_mem_t<BITS>)*n);
    cudaMalloc((void **)&c_mod, sizeof(cgbn_mem_t<BITS>)*1);
    cudaMalloc((void **)&c_r, sizeof(cgbn_mem_t<BITS>)*n);
    param_zz_to_cuda(c_mod,mod);

    uint64_to_cgbn<<<n, MAX_SIZE>>>(c_num,c_result);

    cu_rem<<<(n+ 3 )/ 4 , 128>>>(c_r,c_num,c_pProdnp, c_pProdhnp,c_mod);

    cgbn_mem_t<BITS>* r = new cgbn_mem_t<BITS>[n];
    

    cudaMemcpy(r, c_r, sizeof(cgbn_mem_t<BITS>)*n, cudaMemcpyDeviceToHost);


    
    unsigned char* tmp =  new unsigned char[MAX_SIZE*8];
    for(int i =0;i<n;i++)
    {
        memset(tmp,0,MAX_SIZE*8);
        for(int j = 0; j< MAX_SIZE*2;j++)
        {
            tmp[4*j+0] = (unsigned char)(r[i]._limbs[j]);
            tmp[4*j+1] = (unsigned char)(r[i]._limbs[j]>>8);
            tmp[4*j+2] = (unsigned char)(r[i]._limbs[j]>>16);
            tmp[4*j+3] = (unsigned char)(r[i]._limbs[j]>>24);
        }
        NTL::ZZFromBytes(x[i] ,tmp,MAX_SIZE*8);
    }
 
    delete[] r;
    cudaFree(c_mod);
    cudaFree(c_r);
    cudaFree(c_num);

}

void cuda_reconstruct::reconst(uint64_t* c_result,uint64_t* c_rx)
{  

    //cudaMemcpy(c_rx,rx,np*n*sizeof(uint64_t),cudaMemcpyHostToDevice);
    //cudaMemset(c_result,0,MAX_SIZE*n*sizeof(uint64_t));

    dim3 grids(n,1,1);
    dim3 threads(np,1,1);
    cu_reconst<<<grids,threads>>>(c_result,c_pHatnp, c_rx,c_pVec,c_pHatInvModpnp, c_coeffpinv_arraynp);

}

void cuda_reconstruct::cuda_icrt_run(NTL::ZZ* x,uint64_t* c_rx ,const NTL::ZZ& mod)
{
    uint64_t* c_result;
    cudaMalloc(&c_result,MAX_SIZE*n*sizeof(uint64_t));
    reconst(c_result,c_rx);
    cu_return_data(x,c_result,mod);
    cudaFree(c_result);
}