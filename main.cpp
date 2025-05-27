#include "EnerBe.h"
#include "Mesh.h"
#include "Initialize.h"
#include "MM.h"
#include "Parser.h"
#include <iostream>

template<typename T>
void run_algorithm(int size, AlgorithmType chosen_alg) {
    Mesh2D<T> mesh(size, size);
    Initialize2D<T> initializer(mesh);
    initializer.initializeMatrices();
    MM<T> mm(mesh);
    
    switch (chosen_alg) {
        case AlgorithmType::naiveGEMM:
            mm.naiveGEMM();
            break;
        case AlgorithmType::tiledGEMM:
            mm.tiledGEMM();
            break;
        case AlgorithmType::naiveOpenMPGEMM:
            mm.naiveOpenMPGEMM();
            break;
        case AlgorithmType::tiledOpenMPGEMM:
            mm.tiledOpenMPGEMM();
            break;
        default:
            throw std::runtime_error("Unknown algorithm type");
    }
}

int main(int argc, char *argv[]) {
    int size = 0;
    std::string precision = "single"; // default precision
    AlgorithmType chosen_alg = AlgorithmType::Unknown;

    parse_arguments(argc, argv, size, precision, chosen_alg);
    if (precision == "single") {
        run_algorithm<float>(size, chosen_alg);
    } else if (precision == "double") {
        run_algorithm<double>(size, chosen_alg);
    } else if (precision == "half") {
        run_algorithm<_Float16>(size, chosen_alg);
    } else {
        std::cerr << "Unknown precision: " << precision << std::endl;
        return 1;
    }

    return 0;
}