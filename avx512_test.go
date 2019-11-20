package avx512

import (
	"testing"
)

func TestAdd(t *testing.T) {
	for _, size := range []int{16, 32, 64} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			z := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)
			defer MmFree(z)

			truth := make([]float32, size)
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				truth[i] = x[i] + y[i]
			}

			Add(size, x, y, z)

			for i := 0; i < size; i++ {
				if truth[i] != z[i] {
					t.Errorf("Add should return %f in %d, but %f", truth[i], i, z[i])
				}
			}
		}(size)
	}
}

func TestSub(t *testing.T) {
	for _, size := range []int{16, 32, 64} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			z := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)
			defer MmFree(z)

			truth := make([]float32, size)
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				truth[i] = x[i] - y[i]
			}

			Sub(size, x, y, z)

			for i := 0; i < size; i++ {
				if truth[i] != z[i] {
					t.Errorf("Mul should return %f in %d, but %f", truth[i], i, z[i])
				}
			}
		}(size)
	}
}

func TestDot(t *testing.T) {
	for _, size := range []int{15, 31, 64} {
		func(size int) {
			x := MmMalloc(size)
			y := MmMalloc(size)
			defer MmFree(x)
			defer MmFree(y)

			var truth float32
			for i := 0; i < size; i++ {
				x[i] = float32(i)
				y[i] = float32(i + 1)
				truth += x[i] * y[i]
			}

			result := Dot(size, x, y)
			if truth != result {
				t.Errorf("Dot should return %f, but %f", truth, result)
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
