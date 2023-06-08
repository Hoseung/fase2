#include "../src/HEAAN.h"
#include "../src/SerializationUtils.h"
#include "../src/Scheme.h"

using namespace std;

int main(int argc, char **argv) {

	long logq = 400; ///< Ciphertext Modulus
	long logp = 30; ///< Real message will be quantized by multiplying 2^40
	long logn = 4; ///< log2(The number of slots)

    bool isSerialized = true;

	cout << "!!! START TEST ENCRYPT !!!" << endl;
	srand(time(NULL));
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring, isSerialized);

	long n = (1 << logn);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(n);
	Ciphertext cipher;

	scheme.encrypt(cipher, mvec, n, logp, logq);

	complex<double>* dvec = scheme.decrypt(secretKey, cipher);

	StringUtils::compare(mvec, dvec, n, "val");

	cout << "!!! END TEST ENCRYPT !!!" << endl;
}