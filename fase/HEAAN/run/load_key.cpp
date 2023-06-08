#include "../src/HEAAN.h"
#include "../src/SerializationUtils.h"
#include "../src/Scheme.h"

using namespace std;

int main(int argc, char **argv) {

	long logq = 800; ///< Ciphertext Modulus
	long logp = 30; ///< Real message will be quantized by multiplying 2^40
	long logn = 4; ///< log2(The number of slots)

    bool isSerialized = true;

    Key* key = SerializationUtils::readKey("serkey/ENCRYPTION.txt");

    std::cout << key << std::endl;

    SerializationUtils::writeKey(key, "serkey'test.txt");
}