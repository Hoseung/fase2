
class cu_bignum_handler
{
private:

    long N, np;
public:
    cu_bignum_handler(long _N,long _np);
    ~cu_bignum_handler();
    void uint_to_cgbn(uint64_t* x);
    void test();


};