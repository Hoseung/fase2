#include "../src/HEAAN.h"


int main(int argc, char **argv) {
    Ring ring;
    SecretKey sk(ring);
    Scheme scheme(sk, ring);
    long logq = 800; ///< Ciphertext Modulus. Should be smaller than logQ in "src/Params.h" : 800
    long logp = 40; ///Scaling factor.

    int circuitDepth = 2;
    long logn = 7; // -> 128
    long n = 1<< logn;

	complex<double>* mvec1 = EvaluatorUtils::randomComplexArray(n);
	complex<double>* mvec2 = EvaluatorUtils::randomComplexArray(n);
	complex<double>* mmult = new complex<double>[n];
	for(long i = 0; i < n; i++) {
		mmult[i] = mvec1[i] * mvec2[i];
	}

    Ciphertext c_result, c_multiplicationResult;
    Ciphertext c_fieldVec, cipher2;
    long r = 1;

	scheme.encrypt(c_fieldVec, mvec1, n, logp, logq);
	scheme.encrypt(cipher2, mvec2, n, logp, logq);




    

    for (int j = 0; j < circuitDepth; j ++)
    {
        std::cout << "Multiplication " << j << std::endl;

        //Perform matrix-vector multiplication
        scheme.mult(c_multiplicationResult,c_matrix[0],c_fieldVec);

        //new cM: logq = logq², logp = logp - logq -> rescale, so that it can be added with c_addIdentity
        scheme.reScaleByAndEqual(c_multiplicationResult, logp);

        //No multiplication was performed on c_addIdentity, therefore logq still has the initial value. 
        //In order to add it with cM the
        //logp must match as well.
        scheme.modDownByAndEqual(c_addIdentity, logp);

        //At the end of the process c_fieldVec will get the value from c_result, which has the same logp and logq as cM now.
        //For the next multiplication process they have to match.
        scheme.modDownByAndEqual(c_matrix[0],logp);
        scheme.add(c_result, c_multiplicationResult, c_addIdentity);

        scheme.leftRotateFastAndEqual(c_fieldVec, r);

        for(int i = 1; i< n; i++)
        {
            scheme.mult(c_multiplicationResult,c_matrix[i],c_fieldVec);
            scheme.reScaleByAndEqual(c_multiplicationResult, logp);
            scheme.modDownByAndEqual(c_matrix[i],logp);
            scheme.addAndEqual(c_result, c_multiplicationResult);

            scheme.leftRotateFastAndEqual(c_fieldVec, r);

        }
       }


    return 0;
}
