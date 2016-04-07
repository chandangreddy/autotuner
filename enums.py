class Targets:
    cuda   = "cuda"
    opencl = "opencl"
    prl    = "prl"
    
class Crossover:
    one_point = "one_point"
    two_point = "two_point"

class Compilers:
    gcc     = "gcc"
    gxx     = "g++"
    nvcc    = "nvcc"
    clang   = "clang"
    clangxx = "clang++"
    llvm    = "llvm"
    
class SearchStrategy:
    ga                  = "ga"
    random              = "random"
    simulated_annealing = "simulated-annealing"
    exhaustive          = "exhaustive"

class Status:
    passed = "passed"
    failed = "failed"
    timeout = "timedout"
    ppcgtimeout = "ppcg_timeout"
    
