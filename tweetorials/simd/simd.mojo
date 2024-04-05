import math
from python import Python
from testing import assert_equal, assert_not_equal

def main():
    # Int32 is Scalar[DType.int32]
    a = Int32(42)
    b = Scalar[DType.int32](42)
    assert_equal(a, b)

    # Scalar[DType.int32] is SIMD[DType.int32, 1]
    c = SIMD[DType.int32, 1](42)
    assert_equal(b, c)

    # initialize a SIMD vector of size (power of 2)
    d = SIMD[DType.int32, 4](0, 1, 2, 3)
    e = SIMD[DType.int32, 4]().splat(10)  # [10, 10, 10, 10]
    print(
        d * e, d / e, d % e, d**e
    )  # [0, 10, 20, 30] [0, 0, 0, 0] [0, 1, 2, 3] [0, 1, 1024, 59049]
    # shuffle takes a mask of permutation indices and permutes d accordingly
    print(d.shuffle[1, 3, 2, 0]())  # [1, 3, 2, 0]
    # joining / concatenating two SIMD vectors
    print(d.join(e))  # [0, 1, 2, 3, 10, 10, 10, 10]

    dd = d.cast[DType.bool]()  # [False, True, True, True]
    ee = ~e.cast[DType.bool]()  # [False, False, False, False]
    print(
        dd & ee, dd | ee, dd ^ ee
    )  # [False, False, False, False] [False, True, True, True] [False, False, False, False]

    # interleave combines two SIMD vectors into one by taking
    # one element from the first and another from the second
    # weaving the vectors together
    print(d.interleave(e)) # [0, 10, 1, 10, 2, 10, 3, 10]

    # deinterleave reverses it
    new = d.interleave(e).deinterleave()
    assert_equal(new[0], d)
    assert_equal(new[1], e)

    # dot product using SIMD `__mul__` and `reduce_add`
    def dot_product(x: SIMD, y: SIMD[x.type, x.size]) -> Scalar[x.type]:
        return (x * y).reduce_add()

    assert_equal(dot_product(a, b), 42 * 42)
    assert_equal(dot_product(d, e), 60)

    # absolute value using `select`
    def abs(x: SIMD) -> SIMD[x.type, x.size]:
        # exercise: can this be more efficient
        # depending on `type` i.e. unsigned or floats?
        return (x >= 0).select(x, -x)

    assert_equal(
        abs(SIMD[DType.float32, 4](-1, -2, 0, 1)),
        SIMD[DType.float32, 4](1, 2, 0, 1),
    )

    # softmax using `reduce_max`, `math.exp` and `reduce_add`
    def softmax(x: SIMD) -> SIMD[x.type, x.size]:
        xm = x - x.reduce_max()
        exps = math.exp(xm)
        return exps / exps.reduce_add()

    # vector attention softmax(qk^T)v
    def vector_attention[
        size: Int
    ](
        q: SIMD[DType.float32, size],
        k: SIMD[DType.float32, size],
        v: SIMD[DType.float32, size],
    ) -> SIMD[DType.float32, size]:
        scale_factor = 1 / math.sqrt(len(k))
        return softmax(dot_product(q, k) * scale_factor) * v

    ret = vector_attention(
        SIMD[DType.float32, 4](-4, 0.25, 3.14, 4),
        SIMD[DType.float32, 4](1.1, -0.1, -9, 1),
        SIMD[DType.float32, 4](1, 0.2, 3, 4),
    )
    print(ret)  # [1.0, 0.2000, 3.0, 4.0]

    # compare to torch scaled dot product attention
    torch = Python.import_module("torch")
    ret_pt = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor([-4, 0.25, 3.14, 4]).unsqueeze(0),
        torch.tensor([1.1, -0.1, -9, 1]).unsqueeze(0),
        torch.tensor([1, 0.2, 3, 4]).unsqueeze(0),
    )
    print(ret_pt.squeeze(0))  # tensor([1.0000, 0.2000, 3.0000, 4.0000])
