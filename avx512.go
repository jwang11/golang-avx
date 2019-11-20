package avx512

/*
#cgo CFLAGS: -mavx512f -mavx512vl -mavx512bw -mavx512vnni
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>

int32_t avx2_dot_int8(const size_t n, int8_t *x, int8_t *y)
{
    static const size_t single_size = 32;
    const size_t end = n / single_size;
    const int16_t op4[16] = {[0 ... 15] = 1};
    __m256i *vx = (__m256i *)x;
    __m256i *vy = (__m256i *)y;
    volatile __m256i vsum = {0};
    for(size_t i=0; i<end; ++i) {
      __m256i vresult1 = _mm256_maddubs_epi16(vx[i], vy[i]);
      __m256i vresult2 = _mm256_madd_epi16(vresult1, *(__m256i *)&op4);
      vsum = _mm256_add_epi32(vsum, vresult2);
    }
    int32_t *t = (int32_t *)&vsum;
    volatile int32_t sum = 0;
    for (int i=0; i<8; i++) {
      sum += t[i];
    }
    return sum;
}

int32_t avx512_dot_vnni(const size_t n, int8_t *x, int8_t *y)
{
    static const size_t single_size = 64;
    const size_t end = n / single_size;
    __m512i *vx = (__m512i *)x;
    __m512i *vy = (__m512i *)y;
    volatile __m512i vsum = {0};
    for(size_t i=0; i<end; ++i) {
      vsum = _mm512_dpbusds_epi32(vsum, vx[i], vy[i]);
    }
    int32_t *t = (int32_t *)&vsum;
    volatile int32_t sum = 0;
    for (int i=0; i<16; i++) {
      sum += t[i];
    }
    return sum;

}

int32_t avx512_dot_int8(const size_t n, int8_t *x, int8_t *y)
{
    static const size_t single_size = 64;
    const size_t end = n / single_size;
    const int16_t op4[32] = {[0 ... 31] = 1};
    __m512i *vx = (__m512i *)x;
    __m512i *vy = (__m512i *)y;
    volatile __m512i vsum = {0};
    for(size_t i=0; i<end; ++i) {
      __m512i vresult1 = _mm512_maddubs_epi16(vx[i], vy[i]);
      __m512i vresult2 = _mm512_madd_epi16(vresult1, *(__m512i *)&op4);
      vsum = _mm512_add_epi32(vsum, vresult2);
    }
    int32_t *t = (int32_t *)&vsum;
    volatile int32_t sum = 0;
    for (int i=0; i<16; i++) {
      sum += t[i];
    }
    return sum;
}
*/
import "C"
import (
	"math"
	"reflect"
	"unsafe"
)

func MmMalloc_int8(size int) []int8 {
	size_ := size
	size = align(size)
	ptr := C._mm_malloc((C.size_t)(C.sizeof_int8_t*size), 64)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(ptr)),
		Len:  size,
		Cap:  size,
	}
	goSlice := *(*[]int8)(unsafe.Pointer(&hdr))
	if size_ != size {
		for i := size_; i < size; i++ {
			goSlice[i] = 0
		}
	}
	return goSlice
}

func MmFree_int8(v []int8) {
	C._mm_free(unsafe.Pointer(&v[0]))
}

func Dot_avx512_vnni(size int, x, y []int8) int32 {
	size = align(size)
	dot := C.avx512_dot_vnni((C.size_t)(size), (*C.int8_t)(&x[0]), (*C.int8_t)(&y[0]))
	return int32(dot)
}

func Dot_avx512_int8(size int, x, y []int8) int32 {
	size = align(size)
	dot := C.avx512_dot_int8((C.size_t)(size), (*C.int8_t)(&x[0]), (*C.int8_t)(&y[0]))
	return int32(dot)
}

func Dot_avx2_int8(size int, x, y []int8) int32 {
	size = align(size)
	dot := C.avx2_dot_int8((C.size_t)(size), (*C.int8_t)(&x[0]), (*C.int8_t)(&y[0]))
	return int32(dot)
}

func align(size int) int {
	return int(math.Ceil(float64(size)/8.0) * 8.0)
}
