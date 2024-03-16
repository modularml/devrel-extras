import math
from memory import memset_zero, memcpy
from sys.info import simdwidthof
from algorithm import vectorize
from algorithm.functional import elementwise
from python import Python
from random import rand

struct MojoArray[dtype: DType = DType.float64](Stringable, Sized):
    var _ptr: DTypePointer[dtype]
    var numel: Int
    alias simd_width: Int = simdwidthof[dtype]()

    # Initializers
    fn __init__(inout self, numel: Int):
        self._ptr = DTypePointer[dtype].alloc(numel)
        self.numel = numel
        memset_zero[dtype](self._ptr, numel)
    
    fn __init__(inout self, numel: Int, _ptr: DTypePointer[dtype]):
        self._ptr = _ptr
        self.numel = numel

    fn __init__(inout self, *data: Scalar[dtype]):
        self.numel = len(data)
        self._ptr = DTypePointer[dtype].alloc(len(data))
        for i in range(len(data)):
            self._ptr[i] = data[i]

    @staticmethod
    fn rand(numel: Int)->Self:
        var _ptr = DTypePointer[dtype].alloc(numel)
        rand(_ptr, numel)
        return Self(numel,_ptr)

    fn __copyinit__(inout self, other: Self):
        self._ptr = other._ptr
        self.numel = other.numel
    
    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        return self._ptr.simd_load[1](idx) 
    
    fn __setitem__(self, idx: Int, val: Scalar[dtype]):
        return self._ptr.simd_store[1](idx, val) 
    
    fn __neg__(self)->Self:
        return self._elemwise_scalar_math[math.mul](-1.0)

    fn __mul__(self, other: Self)->Self:
        return self._elemwise_array_math[math.mul](other)
    
    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.mul](s)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __add__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.add](s)
    
    fn __add__(self, other: Self)->Self:
        return self._elemwise_array_math[math.add](other)
       
    fn __radd__(self, s: Scalar[dtype])->Self:
        return self+s

    fn __sub__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.sub](s)

    fn __sub__(self, other: Self)->Self:
        return self._elemwise_array_math[math.sub](other)

    fn __rsub__(self, s: Scalar[dtype])->Self:
        return -self+s

    fn __len__(self) -> Int:
        return self.numel

    fn __le__(self, val: Scalar[dtype]) -> Self:
        var bool_float_array = Self(len(self))
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            var iters = SIMD[dtype, simd_width](0)
            bool_float_array._ptr.simd_store[simd_width](idx, (self._ptr.simd_load[simd_width](idx) <= val).select(iters+1,iters))
        vectorize[elemwise_vectorize, self.simd_width](len(self))
        return bool_float_array

    fn __gt__(self, val: Scalar[dtype]) -> Self:
        var bool_float_array = Self(len(self))
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            var iters = SIMD[dtype, simd_width](0)
            bool_float_array._ptr.simd_store[simd_width](idx, (self._ptr.simd_load[simd_width](idx) > val).select(iters+1,iters))
        vectorize[elemwise_vectorize, self.simd_width](len(self))
        return bool_float_array

    @staticmethod
    fn from_numpy(np_array: PythonObject) raises->Self:
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var numel = int(np_array.shape[0])
        var _ptr = DTypePointer[dtype].alloc(numel)
        memcpy(_ptr, npArrayPtr, numel)
        return Self(numel,_ptr)

    fn to_numpy(self) raises->PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.zeros(len(self))
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_arr.__array_interface__['data'][0].__index__()).value
        )
    )
        memcpy(npArrayPtr, self._ptr, len(self))
        return np_arr

    fn __str__(self)->String:
        var s:String = ""
        s += "["
        for i in range(len(self)):
            if i>0:
                s+=" "
            s+=self._ptr[i]
        s+="]"
        return s

    fn sqrt(self)->Self:
        return self._elemwise_transform[math.sqrt]()

    fn sum(self)->Scalar[dtype]:
        var s = SIMD[dtype, self.simd_width](0)
        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            s += self._ptr.simd_load[self.simd_width](idx)
        vectorize[vectorize_reduce, self.simd_width](len(self))
        return s.reduce_add()
        
    fn __pow__(self, p: Scalar[dtype])->Self:
        return self._elemwise_pow(p)

    fn _elemwise_pow(self, p: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(len(self))
        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, math.pow[dtype,dtype,simd_width](self._ptr.simd_load[simd_width](idx), SIMD[dtype,simd_width].splat(p)))
        vectorize[tensor_scalar_vectorize, simd_width](len(self))
        return new_array

    fn _elemwise_transform[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(len(self))
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx)))
        vectorize[elemwise_vectorize,simd_width](len(self))
        return new_array

    fn _elemwise_array_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(len(self))
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx), other._ptr.simd_load[simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](len(self))
        return new_array

    fn _elemwise_scalar_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(len(self))
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx), SIMD[dtype, simd_width](s)))
        vectorize[elemwise_vectorize, simd_width](len(self))
        return new_array

def main():
    alias N = 1000000

    plt = Python.import_module("matplotlib.pyplot")
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot()
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])

    for i in range(0,N,1000):
        x = MojoArray.rand(i)-0.5
        y = MojoArray.rand(i)-0.5
        r = (x**2 + y**2).sqrt()
        inside = r <= 0.5
        pi = 4.*inside.sum()/i

        ax.cla()
        title = ax.set_title("Mojo ❤️ Pi\n"+r"$\pi$="+str(pi)[:5]+" Iter:"+str(i))
        ax.set_xlim([-0.5,0.5])
        ax.set_ylim([-0.5,0.5])
        title.set_fontsize(24)
        col = inside.to_numpy()
        ax.scatter(x.to_numpy(),y.to_numpy(),4,col)
        plt.pause(0.2)
        ax.set_aspect('equal')
        plt.draw()