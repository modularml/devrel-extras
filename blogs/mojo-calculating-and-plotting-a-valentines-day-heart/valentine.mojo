import max.graph
from memory import memset_zero, memcpy
from sys.info import simdwidthof
from algorithm.functional import vectorize
from python import Python

# updated for Mojo 24.4


# replacing math functions that were removed from the standard library with simple wrappers
fn math_mul[dtype: DType, width: Int](l: SIMD[dtype, width], r:SIMD[dtype, width]) -> SIMD[dtype, width]:
    return l*r

fn math_add[dtype: DType, width: Int](l: SIMD[dtype, width], r:SIMD[dtype, width]) -> SIMD[dtype, width]:
    return l+r

fn math_sub[dtype: DType, width: Int](l: SIMD[dtype, width], r:SIMD[dtype, width]) -> SIMD[dtype, width]:
    return l-r

fn math_abs[dtype: DType, width: Int](l: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return abs(l)

struct MojoArray[dtype: DType = DType.float64](Stringable):
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

    fn __copyinit__(inout self, other: Self):
        self._ptr = other._ptr
        self.numel = other.numel

    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        return self._ptr.load[width=1](idx) 

    fn __neg__(self)->Self:
        return self._elemwise_scalar_math[math_mul](-1.0)

    fn __mul__(self, other: Self)->Self:
        return self._elemwise_array_math[math_mul](other)

    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math_mul](s)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __add__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math_add](s)

    fn __add__(self, other: Self)->Self:
        return self._elemwise_array_math[math_add](other)

    fn __radd__(self, s: Scalar[dtype])->Self:
        return self+s

    fn __sub__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math_sub](s)

    fn __sub__(self, other: Self)->Self:
        return self._elemwise_array_math[math_sub](other)

    fn __rsub__(self, s: Scalar[dtype])->Self:
        return -self+s

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
        var np_arr = np.zeros(self.numel)
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_arr.__array_interface__['data'][0].__index__()).value
        )
    )
        memcpy(npArrayPtr, self._ptr, self.numel)
        return np_arr

    fn __str__(self)->String:
        var s:String = ""
        s += "["
        for i in range(self.numel):
            if i>0:
                s+=" "
            s = s + str(self._ptr[i])
        s = s +"]"
        return s

    fn sqrt(self)->Self:
        return self._elemwise_transform[math.sqrt]()

    fn cos(self)->Self:
        return self._elemwise_transform[math.cos]()

    fn sin(self)->Self:
        return self._elemwise_transform[math.sin]()

    fn abs(self)->Self:
        return self._elemwise_transform[math_abs]()

    fn __pow__(self, p: Scalar[dtype])->Self:
        return self._elemwise_pow(p)

    fn _elemwise_pow(self, p: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)

        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int):
            new_array._ptr.store[width=simd_width](idx, pow(self._ptr.load[width=simd_width](idx), SIMD[dtype,simd_width].splat(p)))

        vectorize[tensor_scalar_vectorize, simd_width](self.numel)
        return new_array

    fn _elemwise_transform[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.store[width=simd_width](idx, func[dtype, simd_width](self._ptr.load[width=simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](self.numel)
        return new_array

    fn _elemwise_array_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.store[width=simd_width](idx, func[dtype, simd_width](self._ptr.load[width=simd_width](idx), other._ptr.load[width=simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](self.numel)
        return new_array

    fn _elemwise_scalar_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.store[width=simd_width](idx, func[dtype, simd_width](self._ptr.load[width=simd_width](idx), SIMD[dtype, simd_width](s)))
        vectorize[elemwise_vectorize, simd_width](self.numel)
        return new_array



fn main() raises:
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")

    var np_arr = np.arange(-2,2,0.01)
    var x = MojoArray.from_numpy(np_arr)

    var fig = plt.figure()
    var ax = fig.add_subplot()
    _ = ax.set_xlim([-3,3])
    _ = ax.set_ylim([-3,3])   

    var a = MojoArray.from_numpy(np.linspace(0,20,100))
    for i in range(a.numel):
        var y = (x**2)**(1/3.) - 0.9*((3.3-(x*x)).sqrt())*(a[i]*3.14*x).sin()
        _ = ax.cla()
        var title = ax.set_title("Mojo ❤️ Python")
        _ = title.set_fontsize(20)
        _ = ax.set_axis_off()
        _ = ax.plot(x.to_numpy(),y.to_numpy(),'r')
        _ = plt.pause(0.1)
        _ = plt.draw()
