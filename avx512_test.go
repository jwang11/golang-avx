package avx512

import (
	"testing"
)

func TestDot_int8(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			x := MmMalloc_int8(size)
			y := MmMalloc_int8(size)
			defer MmFree_int8(x)
			defer MmFree_int8(y)

			var truth int32
			for i := 0; i < size; i++ {
				x[i] = int8(i % 127)
				y[i] = int8((i + 1) % 127)
				truth += int32(x[i]) * int32(y[i])
			}

			result := Dot_int8(size, x, y)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func TestDot_vnni(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			x := MmMalloc_int8(size)
			y := MmMalloc_int8(size)
			defer MmFree_int8(x)
			defer MmFree_int8(y)

			var truth int32
			for i := 0; i < size; i++ {
				x[i] = int8(i % 127)
				y[i] = int8((i + 1) % 127)
				truth += int32(x[i]) * int32(y[i])
			}

			result := Dot_vnni(size, x, y)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func BenchmarkAVX512DotVnni(b *testing.B) {
    size := 1024
	
    x := MmMalloc_int8(size)
    y := MmMalloc_int8(size)
    defer MmFree_int8(x)
    defer MmFree_int8(y)
    for i := 0; i < size; i++ {
        x[i] = int8(i % 127)
        y[i] = int8((i + 1) % 127)
    }
	b.SetBytes(int64(size))
    b.ResetTimer()
	var result int32
    for i := 0; i < b.N; i++ {
		result += Dot_vnni(size, x, y)
    }
}

func BenchmarkAVX512DotInt8(b *testing.B) {
    size := 1024
	
    x := MmMalloc_int8(size)
    y := MmMalloc_int8(size)
    defer MmFree_int8(x)
    defer MmFree_int8(y)
    for i := 0; i < size; i++ {
        x[i] = int8(i % 127)
        y[i] = int8((i + 1) % 127)
    }
	b.SetBytes(int64(size))
    b.ResetTimer()
	var result int32
    for i := 0; i < b.N; i++ {
		result += Dot_int8(size, x, y)
    }
}
