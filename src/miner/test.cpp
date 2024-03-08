#include <iostream>
using std::cout;
using std::endl;

#include <emmintrin.h>
#include <malloc.h>
#include <time.h>
#include <string.h>

#define ENABLE_PREFETCH


#define f_vector    __m128d
#define i_ptr       size_t
inline void swap_block(f_vector *A,f_vector *B,i_ptr L){
    //  To be super-optimized later.

    f_vector *stop = A + L;

    do{
        f_vector tmpA = *A;
        f_vector tmpB = *B;
        *A++ = tmpB;
        *B++ = tmpA;
    }while (A < stop);
}
void transpose_even(f_vector *T,i_ptr block,i_ptr x){
    //  Transposes T.
    //  T contains x columns and x rows.
    //  Each unit is of size (block * sizeof(f_vector)) bytes.

    //Conditions:
    //  - 0 < block
    //  - 1 < x

    i_ptr row_size = block * x;
    i_ptr iter_size = row_size + block;

    //  End of entire matrix.
    f_vector *stop_T = T + row_size * x;
    f_vector *end = stop_T - row_size;

    //  Iterate each row.
    f_vector *y_iter = T;
    do{
        //  Iterate each column.
        f_vector *ptr_x = y_iter + block;
        f_vector *ptr_y = y_iter + row_size;

        do{

#ifdef ENABLE_PREFETCH
            _mm_prefetch((char*)(ptr_y + row_size),_MM_HINT_T0);
#endif

            swap_block(ptr_x,ptr_y,block);

            ptr_x += block;
            ptr_y += row_size;
        }while (ptr_y < stop_T);

        y_iter += iter_size;
    }while (y_iter < end);
}
int main(){

    i_ptr dimension = 4096;
    i_ptr block = 16;

    i_ptr words = block * dimension * dimension;
    i_ptr bytes = words * sizeof(f_vector);

    cout << "bytes = " << bytes << endl;
//    system("pause");

    f_vector *T = (f_vector*)_mm_malloc(bytes,16);
    if (T == NULL){
        cout << "Memory Allocation Failure" << endl;
        system("pause");
        exit(1);
    }
    memset(T,0,bytes);

    //  Perform in-place data transpose
    cout << "Starting Data Transpose...   ";
    clock_t start = clock();
    transpose_even(T,block,dimension);
    clock_t end = clock();

    cout << "Done" << endl;
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << " seconds" << endl;

    _mm_free(T);
    system("pause");
}