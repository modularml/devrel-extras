from max.extensibility import Tensor, empty_tensor
from custom_ops.relu6 import relu6
from testing import assert_equal

alias type = DType.float32

def test_relu6():
    x = empty_tensor[type](StaticIntTuple[1](5))
    x.store(0, Float32(-1))
    x.store(1, Float32(0))
    x.store(2, Float32(1))
    x.store(3, Float32(6))
    x.store(4, Float32(7))

    expected = empty_tensor[type](StaticIntTuple[1](5))
    expected.store(0, Float32(0))
    expected.store(1, Float32(0))
    expected.store(2, Float32(1))
    expected.store(3, Float32(6))
    expected.store(4, Float32(6))

    assert_equal(relu6(x), expected)

# uncomment for testing and run `mojo test_relu6.mojo`
# def main():
#     test_relu6()
