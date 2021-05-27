// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_SPARSE_GEMM_TEST_HPP
#define Sapphire_SPARSE_GEMM_TEST_HPP

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace Sapphire::Test
{
struct PerformanceData
{
    size_t m, n, k;
    float sparsity;
    long naiveDense;
    long cudaDense;
    long naiveSparse;
    long cudaSparse;
    long cuSparse;

    void PrintData() const
    {
        std::cout << "--- Results (time in microseconds) ---" << std::endl;
        std::cout << "* Sparsity : " << sparsity << std::endl;
        std::cout << "* Naive Dense : " << naiveDense << std::endl;
        std::cout << "* Naive Sparse : " << naiveSparse << std::endl;
        std::cout << "* Cuda Dense : " << cudaDense << std::endl;
        std::cout << "* Cuda Sparse (custom) : " << cudaSparse << std::endl;
        std::cout << "* Cuda Sparse (cuSparse) : " << cuSparse << std::endl;
        std::cout << "-------------------------------------\n" << std::endl;
    }

    static void WriteCsvHeader(std::ofstream& filestream)
    {
        std::string dl = ",";
        filestream << "m" << dl << "n" << dl << "k" << dl << "sparsity" << dl
                   << "naiveDense" << dl << "naiveSparse" << dl << "cudaDense"
                   << dl << "cudaSparse" << dl << "cuSparse" << std::endl;
    }

    void WriteCsv(std::ofstream& filestream) const
    {
        std::string dl = ",";
        filestream << std::to_string(m) << dl << std::to_string(n) << dl
                   << std::to_string(k) << dl << std::to_string(sparsity) << dl
                   << std::to_string(naiveDense) << dl
                   << std::to_string(naiveSparse) << dl
                   << std::to_string(cudaDense) << dl
                   << std::to_string(cudaSparse) << dl
                   << std::to_string(cuSparse) << std::endl;
    }
};

void LoadDistTestFixed(bool printVerbose);

void LoadDistTest(bool printVerbose);

long SparseGemmTestComplex(unsigned int m, unsigned int n, unsigned int k,
                           size_t minimumNumMatrices, bool print,
                           bool printVerbose);

long SparseGemmTestSimple(unsigned int m, unsigned int n, unsigned int k,
                          size_t numMatrices, bool print, bool printVerbose);

void SparseTestCorrectnessHost(size_t m, size_t n, size_t k, size_t numMatrices,
                               float sparsity, bool printResult);

void SparseTestCorrectnessCuda(size_t m, size_t n, size_t k, size_t numMatrices,
                               float sparsity, bool printResult);

void SparseMatrixConversionTest(size_t m, size_t n, size_t numMatrices,
                                float sparsity, bool printResult);

PerformanceData PerformanceTest(size_t m, size_t n, size_t k,
                                size_t numMatrices, float sparsity);

}  // namespace Sapphire::Test

#endif  // Sapphire_SPARSEGEMMTEST_HPP
