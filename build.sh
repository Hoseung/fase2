export ROOT_DIR=$PWD

rm build/* -rf
rm fase/HEAAN.cpython*.so

cd $ROOT_DIR/fase/HEAAN/lib/
make clean && make  -j 4

cd $ROOT_DIR
pip install -e .
