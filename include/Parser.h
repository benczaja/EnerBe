#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <stdbool.h> // needed for bool usage


enum class AlgorithmType {
    naiveGEMM,
    tiledGEMM,
    naiveOpenMPGEMM,
    tiledOpenMPGEMM,

    #ifdef ENABLE_CUDA
        threadCudaGEMM,
    #endif
    Unknown
};

inline AlgorithmType algorithm_from_string(const std::string& s) {
    if (s == "--naive-gemm")    return AlgorithmType::naiveGEMM;
    if (s == "--tiled-gemm")    return AlgorithmType::tiledGEMM;
    if (s == "--naiveopenmp-gemm")    return AlgorithmType::naiveOpenMPGEMM;
    if (s == "--tiledopenmp-gemm")    return AlgorithmType::tiledOpenMPGEMM;
    #ifdef ENABLE_CUDA
        if (s == "--threadcuda-gemm") return AlgorithmType::threadCudaGEMM;
    #endif  
    return AlgorithmType::Unknown;
}

inline std::string algorithm_to_string(AlgorithmType alg) {
    switch (alg) {
    case AlgorithmType::naiveGEMM: return "--naive-gemm";
    case AlgorithmType::tiledGEMM: return "--tiled-gemm";
    case AlgorithmType::naiveOpenMPGEMM: return "--naiveopenmp-gemm";    
    case AlgorithmType::tiledOpenMPGEMM: return "--tiledopenmp-gemm";
    #ifdef ENABLE_CUDA
        case AlgorithmType::threadCudaGEMM: return "--threadcuda-gemm";
    #endif
    default:                            return "Unknown";
    }
}


int res;
int rounds=0;

void print_usage()
{
    fprintf(stderr, "Example usage:\n");
    fprintf(stderr, "EnerBe [--type-algorithm] [--precision] (problem size) \n");

    AlgorithmType all_algorithms[] = {
        AlgorithmType::naiveGEMM, AlgorithmType::tiledGEMM,
        AlgorithmType::naiveOpenMPGEMM, AlgorithmType::tiledOpenMPGEMM
        #ifdef ENABLE_CUDA
            , AlgorithmType::threadCudaGEMM
        #endif
        };
    fprintf(stderr, "\nAccepted algorithms are:\n");
    for (auto alg : all_algorithms) {
        std::cout << algorithm_to_string(alg) << std::endl;
    }
    fprintf(stderr, "Accepted precisions are:\n");
    std::cout << "--half (Most likely unsuported on your CPU. check the compiler flags, youll have better luck on AARCH)" << std::endl;
    std::cout << "--single" << std::endl;
    std::cout << "--double" << std::endl;
}


bool isNumber(char number[])
{
    int i = 0;

    //checking for negative numbers
    if (number[0] == '-')
        i = 1;
    for (; number[i] != 0; i++)
    {
        //if (number[i] > '9' || number[i] < '0')
        if (!isdigit(number[i]))
            return false;
    }
    return true;
}


void parse_arguments(int count, char*  args[], int& problem_size, std::string& precision, AlgorithmType& algorithm) {
    bool success_number = false;
    bool success_algo = false;
    bool success_precision = false;

    if (count != 4 ){
        printf("I need an algorithm, problem size, and precision as arguments.\nSee what I accept: ./EnerBe -h \n");
        print_usage();
        exit(1);
    }

    for (int i = 0; i < count; i++) {
        std::string arg = args[i];

        if (arg == "-h") {
            print_usage();
            exit(1);
        }

        if (!success_algo) {
            AlgorithmType alg = algorithm_from_string(arg);
            if (alg != AlgorithmType::Unknown) {
                algorithm = alg;
                success_algo = true;
            }
        }

        if (!success_precision) {
            if (arg == "--float") {
                precision = "float";
                success_precision = true;
            }
            else if (arg == "--half") {
                precision = "half";
                success_precision = true;
            }
            else if (arg == "--double") {
                precision = "double";
                success_precision = true;
            }
        }

        if (!success_number && isNumber(args[i])) {
            sscanf(args[i], "%d", &problem_size);
            success_number = true;
        }
    }

    if (!success_algo) {
        printf("Could not match Algorithm type\n");
        printf("Accepted algorithms are:\n");
        AlgorithmType all_algorithms[] = {
            AlgorithmType::naiveGEMM, AlgorithmType::tiledGEMM,
            AlgorithmType::naiveOpenMPGEMM, AlgorithmType::tiledOpenMPGEMM
        };
        for (auto alg : all_algorithms) {
            std::cout << algorithm_to_string(alg) << std::endl;
        }
        exit(1);
    }
    if (!success_number) {
        printf("Could not read number\n");
        exit(1);
    }
}