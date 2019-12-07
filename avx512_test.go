package avx512

import (
	"testing"
	"math/rand"
)
const benchsize = 1036

func TestDotAvx512Int8(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			vx := Make_int8(size)
			vy := Make_int8(size)

			var truth int32
			for i := 0; i < size; i++ {
        		vx[i] = int8(rand.Intn(127))
        		vy[i] = int8(rand.Intn(127))
				truth += int32(vx[i]) * int32(vy[i])
			}

			result := Dot_avx512_int8(size, vx, vy)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func TestDotAvx512Vnni(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			vx := Make_int8(size)
			vy := Make_int8(size)

			var truth int32
			for i := 0; i < size; i++ {
        		vx[i] = int8(rand.Intn(127))
        		vy[i] = int8(rand.Intn(127))
				truth += int32(vx[i]) * int32(vy[i])
			}

			result := Dot_avx512_vnni(size, vx, vy)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func TestDotAvx2Int8(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			vx := Make_int8(size)
			vy := Make_int8(size)

			var truth int32
			for i := 0; i < size; i++ {
        		vx[i] = int8(rand.Intn(127))
        		vy[i] = int8(rand.Intn(127))
				truth += int32(vx[i]) * int32(vy[i])
			}

			result := Dot_avx2_int8(size, vx, vy)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func BenchmarkAvx2DotInt8(b *testing.B) {
    size := benchsize
	vx := Make_int8(size)
	vy := Make_int8(size)
    for i := 0; i < size; i++ {
        vx[i] = int8(rand.Intn(127))
        vy[i] = int8(rand.Intn(127))
    }
	b.SetBytes(int64(size))
    b.ResetTimer()
	var result int32 = 0
    for i := 0; i < b.N; i++ {
		result += Dot_avx2_int8(size, vx, vy)
		vx[i % size] = int8(result)
		vy[i % size] = int8(result)
    }
}

func BenchmarkAVX512DotInt8(b *testing.B) {
    size := benchsize
	vx := Make_int8(size)
	vy := Make_int8(size)
    for i := 0; i < size; i++ {
        vx[i] = int8(rand.Intn(127))
        vy[i] = int8(rand.Intn(127))
    }
	b.SetBytes(int64(size))
    b.ResetTimer()
	var result int32 = 0
    for i := 0; i < b.N; i++ {
		result += Dot_avx512_int8(size, vx, vy)
		vx[i % size] = int8(result)
		vy[i % size] = int8(result)
    }
}

func BenchmarkAVX512DotVnni(b *testing.B) {
    size := benchsize
	vx := Make_int8(size)
	vy := Make_int8(size)
    for i := 0; i < size; i++ {
        vx[i] = int8(rand.Intn(127))
        vy[i] = int8(rand.Intn(127))
    }
	b.SetBytes(int64(size))
    b.ResetTimer()
	var result int32 = 0
    for i := 0; i < b.N; i++ {
		result += Dot_avx512_vnni(size, vx, vy)
		vx[i % size] = int8(result)
		vy[i % size] = int8(result)
    }
}
