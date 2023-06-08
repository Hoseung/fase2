build_fpga=false
build_cuda=false
build_seal=false
build_palisade=false


export ROOT_DIR=$PWD
export CGBN_INCLUDE_DIR=$ROOT_DIR/third_party/CGBN/include

cd $ROOT_DIR/fase/HEAAN/lib/
make clean && make  -j 4

if [ "$build_fpga" = true ] ; then
    cd $ROOT_DIR/fase/HEAAN_fpga/lib/
    make clean && make -j 4
fi

if [ "$build_cuda" = true ] ; then
    cd $ROOT_DIR/fase/HEAAN_cuda/lib/
    make clean && make -j 4
fi

if [ "$build_seal" = true ] ; then
    cd $ROOT_DIR/fase/SEAL
    rm build/* -rf
    cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF
    cmake --build build -j 4
fi

if [ "$build_palisade" = true ] ; then
    mkdir $ROOT_DIR/fase/bind/palisade/build
    cd $ROOT_DIR/fase/bind/palisade/build
    cmake .. 
    make -j 4
    ln -s lib/pycrypto.so $ROOT_DIR/fase/ 
fi

cd $ROOT_DIR
pip install -e .
