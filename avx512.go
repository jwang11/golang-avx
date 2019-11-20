package avx512

/*
#cgo CFLAGS: -mavx512f -mavx512vl -mavx512bw -mavx512vnni
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <immintrin.h>
void avx_add(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 16;
    const size_t end = n / single_size;
    __m512 *vz = (__m512 *)z;
    __m512 *vx = (__m512 *)x;
    __m512 *vy = (__m512 *)y;
    for(size_t i=0; i<end; ++i) {
      vz[i] = _mm512_add_ps(vx[i], vy[i]);
    }
}

void avx_sub(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 16;
    const size_t end = n / single_size;
    __m512 *vz = (__m512 *)z;
    __m512 *vx = (__m512 *)x;
    __m512 *vy = (__m512 *)y;
    for(size_t i=0; i<end; ++i) {
      vz[i] = _mm512_sub_ps(vx[i], vy[i]);
    }
}

void avx_mul(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 16;
    const size_t end = n / single_size;
    __m512 *vz = (__m512 *)z;
    __m512 *vx = (__m512 *)x;
    __m512 *vy = (__m512 *)y;
    for(size_t i=0; i<end; ++i) {
      vz[i] = _mm512_mul_ps(vx[i], vy[i]);
    }
}

float avx_dot(const size_t n, float *x, float *y)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m512 *vx = (__m512 *)x;
    __m512 *vy = (__m512 *)y;
    __m512 vsum = {0};
    for(size_t i=0; i<end; ++i) {
      vsum = _mm512_add_ps(vsum, _mm512_mul_ps(vx[i], vy[i]));
    }
    __attribute__((aligned(32))) float t[16] = {0};
    _mm512_store_ps(t, vsum);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7] +
	t[8] + t[9] + t[10] + t[11] + t[12] + t[13] + t[14] + t[15];
}

int32_t avx_dot_vnni(const size_t n, int8_t *x, int8_t *y)
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
      printf("%d ", t[i]);
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

func MmMalloc(size int) []float32 {
	size_ := size
	size = align(size)
	ptr := C._mm_malloc((C.size_t)(C.sizeof_float*size), 64)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(ptr)),
		Len:  size,
		Cap:  size,
	}
	goSlice := *(*[]float32)(unsafe.Pointer(&hdr))
	if size_ != size {
		for i := size_; i < size; i++ {
			goSlice[i] = 0.0
		}
	}
	return goSlice
}

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

func MmFree(v []float32) {
	C._mm_free(unsafe.Pointer(&v[0]))
}

func MmFree_int8(v []int8) {
	C._mm_free(unsafe.Pointer(&v[0]))
}

func Add(size int, x, y, z []float32) {
	size = align(size)
	C.avx_add((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]), (*C.float)(&z[0]))
}

func Mul(size int, x, y, z []float32) {
	size = align(size)
	C.avx_mul((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]), (*C.float)(&z[0]))
}

func Sub(size int, x, y, z []float32) {
	size = align(size)
	C.avx_sub((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]), (*C.float)(&z[0]))
}

func Dot(size int, x, y []float32) float32 {
	size = align(size)
	dot := C.avx_dot((C.size_t)(size), (*C.float)(&x[0]), (*C.float)(&y[0]))
	return float32(dot)
}

func Dot_vnni(size int, x, y []int8) int32 {
	size = align(size)
	dot := C.avx_dot_vnni((C.size_t)(size), (*C.int8_t)(&x[0]), (*C.int8_t)(&y[0]))
	return int32(dot)
}

func align(size int) int {
	return int(math.Ceil(float64(size)/8.0) * 8.0)
}
