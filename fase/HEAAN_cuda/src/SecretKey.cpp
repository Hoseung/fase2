/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#include "SecretKey.h"
#include <iostream>
#include <fstream>

SecretKey::SecretKey(Ring& ring) {
	ring.sampleHWT(sx);
}


SecretKey::SecretKey(Ring& ring, std::string path) {
	long* indices = new long[h];
	long* values  = new long[h];	
	ring.sampleHWT(sx, indices, values);

	// for (int i = 0; i<h;i++){ cout << indices[i] <<  " " << values[i] << endl; }

	fstream fout;
	fout.open(path, ios::binary|ios::out);
	fout.write(reinterpret_cast<char*>(indices), h*sizeof(long));
	fout.write(reinterpret_cast<char*>(values), h*sizeof(long));
	fout.close();
	cout << "saving secret key done." << endl;	
}

//void SecretKey::loadKey(std::string path) {
SecretKey::SecretKey(std::string path) {    
	long* indices = new long[h];
	long* values  = new long[h];

	fstream fin;
	fin.open(path, ios::binary|ios::in);
	fin.read(reinterpret_cast<char*>(indices), h*sizeof(long));
	fin.read(reinterpret_cast<char*>(values), h*sizeof(long));
	fin.close();		

	for(int i=0; i<N; i++){ sx[i] = 0; }

	for(int i = 0; i<h;i++){ 
		// cout << indices[i] <<  " " << values[i] << endl; 
		if (values[i] == 1){
			sx[indices[i]] = ZZ(1);
		} else if (values[i] == -1) { 
			sx[indices[i]] = ZZ(-1);
		} else { }
	}
	cout << "loading secret key done." << endl;	

}
