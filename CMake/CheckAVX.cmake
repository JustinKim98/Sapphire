# Check for the presence of AVX and figure out the flags to use for it.
macro(CHECK_FOR_AVX)
    set(AVX_FLAGS)

    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS)
    
    # Check AVX
    check_cxx_source_runs("
        #include <immintrin.h>
        int main()
        {
          __m256 a, b, c;
          const float src[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
          float dst[8];
          a = _mm256_loadu_ps( src );
          b = _mm256_loadu_ps( src );
          c = _mm256_add_ps( a, b );
          _mm256_storeu_ps( dst, c );
          for( int i = 0; i < 8; i++ ){
            if( ( src[i] + src[i] ) != dst[i] ){
              return -1;
            }
          }
          return 0;
        }"
        HAVE_AVX_EXTENSIONS)

    # Check AVX2
    check_cxx_source_runs("
        #include <immintrin.h>
        int main()
        {
          __m256i a, b, c;
          const int src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
          int dst[8];
          a =  _mm256_loadu_si256( (__m256i*)src );
          b =  _mm256_loadu_si256( (__m256i*)src );
          c = _mm256_add_epi32( a, b );
          _mm256_storeu_si256( (__m256i*)dst, c );
          for( int i = 0; i < 8; i++ ){
            if( ( src[i] + src[i] ) != dst[i] ){
              return -1;
            }
          }
          return 0;
        }"
        HAVE_AVX2_EXTENSIONS)

    # Check AVX512
    check_cxx_source_runs("
        #include <immintrin.h>
        int main()
        {
          __m512i a, b, c;
          const uint64_t src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
          int dst[8];
          a =  _mm256_loadu_si512( (__m512i*)src );
          b =  _mm256_loadu_si512( (__m512i*)src );
          c = _mm512_add_epi64( a, b );
          _mm512_storeu_si512( (__m512i*)dst, c );
          for( int i = 0; i < 8; i++ ){
            if( ( src[i] + src[i] ) != dst[i] ){
              return -1;
            }
          }
          return 0;
        }"
        HAVE_AVX512_EXTENSIONS)

        if(HAVE_AVX_EXTENSIONS AND HAVE_AVX2_EXTENSIONS)
            message(STATUS "Found AVX/AVX2 instruction sets")
        endif()
        if(HAVE_AVX512_EXTENSIONS)
            message(STATUS "Found AVX512 instruction set")
        endif()

endmacro(CHECK_FOR_AVX)