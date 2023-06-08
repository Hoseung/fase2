# Install Boost::Python

mkdir deps 
cd deps 

wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz ./

tar -xf boost_1_77_0.tar.gz 

cd boost_1_77_0/

./bootstrap.sh

sudo ./b2 install 

