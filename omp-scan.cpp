#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scanning array A and writing result into prefix_sum array;

void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}



void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n==0) return;
  int tid, chunk_sz, nthreads;
  prefix_sum[0] = 0;
  long* local_sum;
  
  

  #pragma omp parallel private(tid) //tid should be a private variable for the threads.
  {

    tid = omp_get_thread_num();
    
    #pragma omp single
    {

      nthreads = omp_get_num_threads();
      chunk_sz = ceil(n / (double) nthreads);
      printf("No. of threads = %d and the chunk size = %d\n", nthreads, chunk_sz);
      local_sum = new long[nthreads+1];
      local_sum[0] = 0;
    }
    //#pragma omp barrier
    // Calculate the individual sum and store the local results.
    #pragma omp for schedule(static, 1) nowait
    for(int thrd = 0; thrd < nthreads; thrd++) {
      long start = thrd*chunk_sz;
      long end = n < (start+chunk_sz) ? n : (start + chunk_sz);
      for(long i = start+1; i < end; i++) {
        prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      }
      local_sum[thrd+1] = prefix_sum[end-1] + A[end-1];
    }
    #pragma omp barrier   // wait for all the threads to parallelly scan and save their individual sum.

    // get the offset from the partial sum of the other threads.
    long off = 0;
    for(int thrd = 0; thrd <= tid; thrd++){
      off += local_sum[thrd];
    }

    // Now adding the offset to get final value. This step can be parallelized as well which we have done below.
    #pragma omp for schedule(static, 1) nowait
    for(int thrd = 0; thrd < nthreads; thrd++) {
      long start = thrd*chunk_sz;
      long end = n < (start+chunk_sz) ? n : (start + chunk_sz);
      for(long i = start; i < end; i++)
        prefix_sum[i] += off;
    }

    #pragma omp barrier

    #pragma omp single
    {
    delete [] local_sum;
    }
    
  }

}

void print_result(long* b, long n) {
  for(int i = 0; i < n; i++)
    printf("%ld\n", b[i]);
  printf("\n");
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  //omp_set_dynamic(0);
  omp_set_num_threads(1);
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}