import fase
def load():
    if fase.USE_FPGA:
        import fase.HEAAN_fpga as _he
        print("Using FPGA version HEAAN")
    elif fase.USE_CUDA:
        import fase.HEAAN_cuda as _he
        print("Using CUDA version HEAAN")
    else:
        import fase.HEAAN as _he
        print("Using CPU version HEAAN")

    return _he

he = load()
# only import submodules after setting USE_FPGA
#import fase.core
#import fase.cifar
#import fase.RF
#fase.he = load()