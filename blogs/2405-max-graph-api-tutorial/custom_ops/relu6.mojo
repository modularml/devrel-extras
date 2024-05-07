import math
from max.extensibility import Tensor, empty_tensor
from max import register

@register.op("relu6")
fn relu6[type: DType, rank: Int](x: Tensor[type, rank]) -> Tensor[type, rank]:
    var output = empty_tensor[type](x.shape)

    @always_inline
    @parameter
    fn _relu6[width: Int](i: StaticIntTuple[rank]) -> SIMD[type, width]:
        var val = x.simd_load[width](i)
        return math.min(math.max(0, val), 6)

    output.for_each[_relu6]()
    return output^
