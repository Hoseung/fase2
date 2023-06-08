
#include <NTL/ZZ.h>


class cu_remainder_build
{
private:
    //long* c_p;
    //uint64_t* c_p;


    long np;
    long n;
public:
    bool is_table_maked;
    //uint64_t* c_result;
    //_ntl_general_rem_one_struct* red_ss;

    cu_remainder_build(long _n,long _np);

    ~cu_remainder_build();

    uint64_t* cuda_tbl_build(uint64_t* pVec,cudaStream_t stream);

    //void cuda_ZZ_to_uint(NTL::ZZ* data,long pi);
    uint64_t* host_uint_to_cuda_remainder(uint64_t* data, bool* neg_sign, uint64_t* c_p,uint64_t* c_tbl,cudaStream_t stream);

    //void cuda_rem_calc();

    uint64_t* cuda_result_to_host(uint64_t* result);

};