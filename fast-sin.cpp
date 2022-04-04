#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
/* Modified the sin4_intrin function. */

// coefficients in the Taylor series expansion of sin(x)
static constexpr double c2  = -1/(((double)2));
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void sin4_taylor(double* sinx, const double* x) {
  
  for (int i = 0; i < 4; i++) {
    bool sin = true;
    double ans_sgn = 1;
    double sgn = 1;
    //printf("x[%d]: %f\n", i, x[i]);
    double x1  = x[i];
    if(x1<0) sgn = -1;
    x1 = sgn * fmod(fabs(x[i]), (double)(2*M_PI));
    if(fabs(x1) >= M_PI){
      sgn = sgn * -1;
      x1 = sgn * (fabs(x1)-M_PI);
    }
    if(fabs(x1) >= M_PI/2){
      x1 = sgn * (M_PI - fabs(x1));
    }
    if(fabs(x1) > M_PI/4){
      x1 = sgn * (M_PI/2 - fabs(x1));
      if(x1<0) ans_sgn = -1;
      sin = false;
    }
    //if(!sin) printf("cos ");
    //printf("new x: %f\n", x1);
    double x2  = x1 * x1;
   
    if(sin){
      double x3  = x1 * x2;
      double x5  = x3 * x2;
      double x7  = x5 * x2;
      double x9  = x7 * x2;
      double x11 = x9 * x2;
      double s = x1;
      s += x3  * c3;
      s += x5  * c5;
      s += x7  * c7;
      s += x9  * c9;
      s += x11 * c11;
      sinx[i] = s;
    }
    else{
      double x4  = x2 * x2;
      double x6  = x4 * x2;
      double x8  = x6 * x2;
      double x10 = x8 * x2;

      double s = 1;
      s += x2  * c2;
      s += x4  * c4;
      s += x6  * c6;
      s += x8  * c8;
      s += x10 * c10;
      sinx[i] = ans_sgn * s;
      //printf("sin x = %f %f %f\n",sinx[i], ans_sgn, s);
    }
  }
}

void sin4_intrin(double* sinx, const double* x) {
  double* temp_x = (double*) aligned_malloc(4*sizeof(double));
  double* sin = (double*) aligned_malloc(4*sizeof(double));
  double* cos = (double*) aligned_malloc(4*sizeof(double));
  double* ans_sgn = (double*) aligned_malloc(4*sizeof(double));
  
  for (int i = 0; i < 4; i++) {
    sin[i] = 1;
    cos[i] = 0;
    ans_sgn[i] = 1;
    double sgn = 1;
    //printf("x[%d]: %f\n", i, x[i]);
    temp_x[i]  = x[i];
    if((temp_x[i])<0) sgn = -1;
    temp_x[i] = sgn * fmod(fabs(temp_x[i]), (double)(2*M_PI));
    if(fabs(temp_x[i]) >= M_PI){
      sgn = sgn * -1;
      temp_x[i] = sgn * (fabs(temp_x[i])-M_PI);
    }
    if(fabs(temp_x[i]) >= M_PI/2){
      temp_x[i] = sgn * (M_PI - fabs(temp_x[i]));
    }
    if(fabs(temp_x[i]) > M_PI/4){
      temp_x[i] = sgn * (M_PI/2 - fabs(temp_x[i]));
      if(temp_x[i]<0) ans_sgn[i] = -1;
      sin[i] = 0;
      cos[i] = 1;
    }
    
    //if(!sin[i]) printf("cos ");
    //printf("new x: %f\n", temp_x[i]);
  }
  
    // The definition of intrinsic functions can be found at:
    // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
  #if defined(__AVX__)
  
    __m256d x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11;
    __m256d ans_sign = _mm256_load_pd(ans_sgn);
    __m256d sin_v = _mm256_load_pd(sin);
    __m256d cos_v = _mm256_load_pd(cos);
    
    x1  = _mm256_load_pd(temp_x);
    x2  = _mm256_mul_pd(x1, x1);
    x3  = _mm256_mul_pd(_mm256_mul_pd(x1, x2),sin_v);
    x4  = _mm256_mul_pd(_mm256_mul_pd(x2, x2),cos_v);
    x5  = _mm256_mul_pd(_mm256_mul_pd(x3, x2),sin_v);
    x6  = _mm256_mul_pd(_mm256_mul_pd(x4, x2),cos_v);
    x7  = _mm256_mul_pd(_mm256_mul_pd(x5, x2),sin_v);
    x8  = _mm256_mul_pd(_mm256_mul_pd(x6, x2),cos_v);
    x9  = _mm256_mul_pd(_mm256_mul_pd(x7, x2),sin_v);
    x10  = _mm256_mul_pd(_mm256_mul_pd(x8, x2),cos_v);
    x11  = _mm256_mul_pd(_mm256_mul_pd(x9, x2),sin_v);
    x2 = _mm256_mul_pd(x2, cos_v);
    x1 = _mm256_mul_pd(x1, sin_v);
    
    __m256d ones = {(double) 1,(double) 1, (double) 1, (double) 1 };

    __m256d s = x1;
    s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
    s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 )));
    s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
    s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
    s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 ))); 
    s = _mm256_add_pd(s, _mm256_mul_pd(ones, cos_v));
    s = _mm256_add_pd(s, _mm256_mul_pd(x2 , _mm256_set1_pd(c2 )));
    s = _mm256_add_pd(s, _mm256_mul_pd(x4 , _mm256_set1_pd(c4 )));
    s = _mm256_add_pd(s, _mm256_mul_pd(x6 , _mm256_set1_pd(c6 )));
    s = _mm256_add_pd(s, _mm256_mul_pd(x8 , _mm256_set1_pd(c8 ))); 
    s = _mm256_add_pd(s, _mm256_mul_pd(x10 , _mm256_set1_pd(c10 )));
    s = _mm256_mul_pd(s, ans_sign);
    _mm256_store_pd(sinx, s);
    
    //printf("%d %d %d %d %d\n",sinx[0],sinx[1],sinx[2],sinx[3],sinx[1000]);
    aligned_free(temp_x); aligned_free(sin); aligned_free(cos); aligned_free(ans_sgn);
#elif defined(__SSE2__)
  
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3, x5, x7, x9, x11;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;
  x5  = x3 * x2;
  x7  = x5 * x2;
  x9  = x7 * x2;
  x11 = x9 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s += x5  * c5 ;
  s += x7  * c7 ;
  s += x9  * c9 ;
  s += x11  * c11 ;

  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));
  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
  }

  tt.tic();
  long max_rep = 1000;
  for (long rep = 0; rep < max_rep; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
    //break;
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < max_rep; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
    //break;
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  tt.tic();
  for (long rep = 0; rep < max_rep; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
    //break;
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
}
