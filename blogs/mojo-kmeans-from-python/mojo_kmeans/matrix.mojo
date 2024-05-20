from math import mul, div, mod, add, trunc, align_down, align_down_residual
from memory import memset_zero, memcpy
from sys.info import simdwidthof
from algorithm import vectorize
from algorithm.functional import elementwise
from algorithm.reduction import max, min, sum, cumsum, mean, argmin
from random import rand
from sys.intrinsics import strided_load
from python import Python
import benchmark
from benchmark import Unit
from testing import assert_true, assert_equal
from buffer import Buffer, NDBuffer
from buffer.list import DimList

struct Matrix[dtype: DType = DType.float64](Stringable, CollectionElement, Sized):
    var _matPtr: DTypePointer[dtype]
    var rows: Int
    var cols: Int
    alias simd_width: Int = 4*simdwidthof[dtype]()

    @always_inline  
    fn __init__(inout self, rows: Int, cols:Int):
        self._matPtr = DTypePointer[dtype].alloc(rows*cols)
        self.rows = rows
        self.cols = cols
        memset_zero[dtype](self._matPtr, self.rows*self.cols)

    fn __init__(inout self, rows: Int, cols:Int, val: Scalar[dtype]):
        alias simd_width = self.simd_width
        self._matPtr = DTypePointer[dtype].alloc(rows*cols)
        self.rows = rows
        self.cols = cols
        @parameter
        fn splat_val[simd_width: Int](idx: Int) -> None:
            self._matPtr.store[width=simd_width](idx, self._matPtr.load[width=simd_width](idx).splat(val))
        vectorize[splat_val, simd_width](self.rows*self.cols)

    @always_inline
    fn __init__(inout self, other: Self):
        self.rows = other.rows
        self.cols = other.cols
        self._matPtr = DTypePointer[dtype].alloc(len(other))
        memcpy(self._matPtr, other._matPtr, len(other))

    @always_inline  
    fn __init__(inout self, elems: Int):
        self._matPtr = DTypePointer[dtype].alloc(elems)
        self.rows = 1
        self.cols = elems
        memset_zero[dtype](self._matPtr, self.rows*self.cols)

    @always_inline  
    fn __init__(inout self, rows: Int, cols:Int, _matPtr: DTypePointer[dtype]):
        self._matPtr = _matPtr
        self.rows = rows
        self.cols = cols

    @always_inline
    fn __init__(inout self, rows: Int, cols:Int, *data: Scalar[dtype]):
        var data_len = len(data)
        self.rows = rows
        self.cols = cols
        self._matPtr = DTypePointer[dtype].alloc(data_len)
        for i in range(data_len):
            self._matPtr[i] = data[i]

    @always_inline
    fn __init__(inout self, rows: Int, cols:Int, owned list: List[Scalar[dtype]]):
        var list_len = len(list)
        self.rows = rows
        self.cols = cols
        self._matPtr = DTypePointer[dtype].alloc(list_len)
        for i in range(list_len):
            self._matPtr[i] = list[i]

    @always_inline
    fn __init__(inout self, dims: StaticIntTuple[2], vals: List[Scalar[dtype]]):
        var list_len = len(vals)
        self.rows = dims[0]
        self.cols = dims[1]
        self._matPtr = DTypePointer[dtype].alloc(list_len)
        for i in range(list_len):
            self._matPtr[i] = vals[i]

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.rows = other.rows
        self.cols = other.cols
        self._matPtr = DTypePointer[dtype].alloc(self.rows*self.cols)
        memcpy(self._matPtr, other._matPtr, self.rows*self.cols)

    fn __moveinit__(inout self, owned existing: Self):
        self._matPtr = existing._matPtr
        self.rows = existing.rows
        self.cols = existing.cols
        existing.rows = 0
        existing.cols = 0
        existing._matPtr = DTypePointer[dtype]()

    fn __del__(owned self):
        self._matPtr.free()

    fn __setitem__(inout self, elem: Int, val: Scalar[dtype]):
        self._matPtr.store[width=1](elem, val)

    fn __setitem__(inout self, row: Int, col:Int, val: SIMD[dtype,1]):
        return self._matPtr.store[width=1](row*self.cols+col, val)

    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[dtype, 1]:
        return self._matPtr.load(idx) 

    @always_inline
    fn __getitem__(self, x: Int, y: Int) -> SIMD[dtype,1]:
        return self._matPtr.load(x * self.cols + y)

    @always_inline
    fn __getitem__(self, owned row_slice: Slice, col: Int) -> Self:
        return self.__getitem__(row_slice, slice(col,col+1))

    @always_inline
    fn __getitem__(self, row: Int, owned col_slice: Slice) -> Self:
        return self.__getitem__(slice(row,row+1),col_slice)

    @always_inline
    fn __getitem__(self, owned row_slice: Slice, owned col_slice: Slice) -> Self:
        self._adjust_row_slice_(row_slice)
        self._adjust_col_slice_(col_slice)

        var src_ptr = self._matPtr
        var dest_mat = Self(row_slice.__len__(),col_slice.__len__())

        alias simd_width: Int = self.simd_width

        for idx_rows in range(row_slice.__len__()):
            src_ptr = self._matPtr.offset(row_slice[idx_rows]*self.cols+col_slice[0])
            @parameter
            fn slice_col_vectorize[simd_width: Int](idx: Int) -> None:
                dest_mat._matPtr.store[width=simd_width](idx+idx_rows*col_slice.__len__(),src_ptr.simd_strided_load[width=simd_width](col_slice.step))
                src_ptr = src_ptr.offset(simd_width*col_slice.step)
            vectorize[slice_col_vectorize, simd_width](col_slice.__len__())
        return dest_mat

    @always_inline
    fn _adjust_row_slice_(self, inout span: Slice):
        if span.start < 0:
            span.start = self.rows + span.start
            
        if not span._has_end():
            span.end = self.rows
        elif span.end < 0:
            span.end = self.rows+ span.end
        if span.end > self.rows:
            span.end = self.rows

    fn _adjust_col_slice_(self, inout span: Slice):
        if span.start < 0:
            span.start = self.cols + span.start
        if not span._has_end():
            span.end = self.cols
        elif span.end < 0:
            span.end = self.cols + span.end
        if span.end > self.cols:
            span.end = self.cols

    @always_inline
    fn __len__(self) -> Int:
        return self.rows*self.cols

    @always_inline
    fn __mul__(self, mat: Self)->Self:
        return self._elemwise_tensor_tensor[mul](mat)

    @always_inline
    fn __add__(self, mat: Self)->Self:
        return self._elemwise_tensor_tensor[add](mat)

    @always_inline
    fn __add__(self, val: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.add](val)

    @always_inline
    fn __iadd__(inout self, other: Self):
        alias simd_width: Int = self.simd_width
        @parameter
        fn tensor_tensor_vectorize[simd_width: Int](idx: Int) -> None:
            self._matPtr.store[width=simd_width](idx, self._matPtr.load[width=simd_width](idx) + other._matPtr.load[width=simd_width](idx))
        vectorize[tensor_tensor_vectorize, simd_width](len(self))

    @always_inline
    fn __truediv__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.div](s)

    @always_inline
    fn __truediv__(self, mat: Self)->Self:
        return self._elemwise_tensor_tensor[div](mat)

    @always_inline
    fn __pow__(self, p: Int)->Self:
        return self._elemwise_pow(p)

    @always_inline
    fn __eq__(self, val: Int) -> Self:
        alias simd_width = self.simd_width
        var bool_float_array = Self(len(self))
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            var iters = SIMD[dtype, simd_width](0)
            bool_float_array._matPtr.store[width=simd_width](idx, (self._matPtr.load[width=simd_width](idx) == val).select(iters+1,iters))
        vectorize[elemwise_vectorize, self.simd_width](len(self))
        return bool_float_array

    @always_inline
    fn __itruediv__(inout self, s: Scalar[dtype]):
        alias simd_width = self.simd_width
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            self._matPtr.store[width=simd_width](idx, math.div[dtype, simd_width](self._matPtr.load[width=simd_width](idx), SIMD[dtype, simd_width](s)))
        vectorize[elemwise_vectorize, simd_width](len(self))

    fn _transpose_order(self) -> DTypePointer[dtype]:
        var newPtr = DTypePointer[dtype].alloc(self.rows*self.cols)
        alias simd_width: Int = self.simd_width
        for idx_col in range(self.cols):
            var tmpPtr = self._matPtr.offset(idx_col)
            @parameter
            fn convert[simd_width: Int](idx: Int) -> None:
                newPtr.store[width=simd_width](idx+idx_col*self.rows, tmpPtr.simd_strided_load[width=simd_width](self.cols))
                tmpPtr = tmpPtr.offset(simd_width*self.cols)
            vectorize[convert, simd_width](self.rows)
        return newPtr

    fn center(self) -> Self:
        alias simd_width = self.simd_width
        var centered_data = Self(self.rows, self.cols)
        var mean_array = self.mean[axis=0]()
        print(mean_array)
        for idx_mat_row in range(self.rows):
            @parameter
            fn center[simd_width: Int](idx: Int) -> None:
                centered_data._matPtr.store[width=simd_width](idx_mat_row*self.cols+idx, self._matPtr.load[width=simd_width](idx_mat_row*self.cols+idx)-mean_array._matPtr.load[width=simd_width](idx))
            vectorize[center, simd_width](len(mean_array))
        return centered_data

    @always_inline
    fn transpose(self) -> Matrix[dtype]:
        return Matrix[dtype](self.cols, self.rows, self._transpose_order())

    @always_inline
    fn fill_val(inout self, val: SIMD[dtype,1])->None:
        alias simd_width = self.simd_width
        @parameter
        fn splat_val[simd_width: Int](idx: Int) -> None:
            self._matPtr.store[width=simd_width](idx, self._matPtr.load[width=simd_width](idx).splat(val))
        vectorize[splat_val, simd_width](self.rows*self.cols)

    @always_inline
    fn zero(inout self)->None:
        memset_zero[dtype](self._matPtr, len(self))

    @always_inline
    fn sum(self) -> Scalar[dtype]:
        var buf = Buffer[dtype](self._matPtr, len(self))
        return sum(buf)

    @always_inline
    fn min[axis: Int = 1](self) -> Self:
        var new_mat = Self(self.rows,1)
        var self_buf = NDBuffer[dtype,2](self._matPtr, DimList(self.rows, self.cols))
        var new_mat_buf = NDBuffer[dtype,2](new_mat._matPtr, DimList(new_mat.rows))
        min[reduce_axis=axis](self_buf, new_mat_buf)
        return new_mat

    @always_inline
    fn sqrt(self)->Self:
        alias simd_width = self.simd_width
        var new_mat = Self(self.rows,self.cols)
        @parameter
        fn wrapper[simd_width: Int, rank: Int = 1](idx: StaticIntTuple[rank]):
            new_mat._matPtr.store[width=simd_width](idx[0], math.sqrt(self._matPtr.load[width=simd_width](idx[0])))
        elementwise[wrapper, simd_width, 1](len(self))
        return new_mat

    @always_inline
    fn mean[axis: Int = 1](self) -> Self:
        var new_mat: Self
        if axis==0:
            new_mat = Self(self.rows)
        else:
            new_mat = Self(self.cols)
        var self_buf = NDBuffer[dtype,2](self._matPtr, DimList(self.rows, self.cols))
        var new_mat_buf = NDBuffer[dtype,2](new_mat._matPtr, DimList(len(new_mat)))
        mean[reduce_axis=axis](self_buf, new_mat_buf)
        return new_mat

    @always_inline
    fn mean(self, where: Matrix[DType.index]) -> Self:
        var dest_mat = Self(1, self.cols)
        var counter = 0.0
        for idx in range(self.rows):
            if int(where[idx]) == 1:
                dest_mat += self[idx,:]
                counter+=1
        return dest_mat/counter

    @always_inline
    fn argmin(self, axis:Int=1) raises -> Matrix[DType.index]:
        var labels_mat = Matrix[DType.index](self.rows,1)
        var self_buf = NDBuffer[dtype,2](self._matPtr, DimList(self.rows, self.cols))
        var labels_mat_buf = NDBuffer[DType.index,2](labels_mat._matPtr, DimList(labels_mat.rows, 1))
        argmin(self_buf, -1, labels_mat_buf)
        return labels_mat

    @always_inline
    fn cumsum(self) -> Self:
        var cumsum_mat = Self(self.rows,self.cols)
        var self_buf = Buffer[self.dtype](self._matPtr, len(self))
        var cumsum_buf = Buffer[self.dtype](cumsum_mat._matPtr, len(self))
        cumsum(cumsum_buf, self_buf)
        return cumsum_mat

    @staticmethod
    fn rand(*dims: Int)->Self:
        var _matPtr = DTypePointer[dtype].alloc(dims[0] * dims[1])
        rand(_matPtr, dims[0] * dims[1])
        return Self(dims[0],dims[1],_matPtr)

    @staticmethod
    fn from_numpy(np_array: PythonObject) raises -> Self:
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var rows = int(np_array.shape[0])
        var cols = int(np_array.shape[1])
        var _matPtr = DTypePointer[dtype].alloc(rows*cols)
        memcpy(_matPtr, npArrayPtr, rows*cols)
        var out = Self(rows,cols,_matPtr)
        return out ^

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.zeros((self.rows,self.cols))
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_arr.__array_interface__['data'][0].__index__()).value
        )
    )
        memcpy(npArrayPtr, self._matPtr, len(self))
        return np_arr ^

    fn print_linear(self):
        print("[",end="")
        for i in range(self.rows*self.cols):
            if i>0:
                print(" ",end="")
            print(self._matPtr[i],end="")
        print("]\n")

    fn __str__(self) -> String:        
        var rank:Int = 2
        var prec:Int = 4
        var printStr:String = ""
        if self.rows == 1:
            rank = 1
        var rows:Int=0
        var cols:Int=0
        if rank==0 or rank>3:
            print("Error: Tensor rank should be: 1,2, or 3. Tensor rank is ", rank)
            return ""
        if rank==1:
            rows = 1
            cols = self.cols
        if rank==2:
            rows = self.rows
            cols = self.cols
        var val:Scalar[dtype]=0.0
        var ctr: Int = 0
        printStr+=""
        for i in range(rows):
            if rank>1:
                if i==0:
                    printStr+="["
                else:
                    printStr+="\n "
            printStr+="["
            for j in range(cols):  
                if rank==1:
                    val = self[j]
                if rank==2:
                    val = self[i,j]
                if dtype != DType.bool and dtype != DType.index:
                    var int_str: String
                    if val >= 0.0:
                        int_str = " "+str(trunc(val).cast[DType.index]())
                    else:
                        # val = math.abs(val)
                        int_str = str(trunc(val).cast[DType.index]())
                    var float_str: String = ""
                    if mod(val,1)==0:
                        float_str = "0"
                    else:
                        try:
                            float_str = str(mod(val,1)).split('.')[-1][0:4]
                        except:
                            return ""
                    var s: String = int_str+"."+float_str
                    if j==0:
                        printStr+=s
                    else:
                        printStr+="  "+s
                else:
                    if j==0:
                        printStr+=str(val)
                    else:
                        printStr+="  "+str(val)
            printStr+="]"
        if rank>1:
            printStr+="]"
        printStr+="\n"
        if rank>2:
            printStr+="]"
        printStr+="Matrix: "+str(self.rows)+'x'+str(self.cols)+" | "+"DType:"+str(dtype)+"\n"
        return printStr

    @always_inline
    fn _elemwise_scalar_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = self.simd_width
        var new_mat = Self(len(self))
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_mat._matPtr.store[width=simd_width](idx, func[dtype, simd_width](self._matPtr.load[width=simd_width](idx), SIMD[dtype, simd_width](s)))
        vectorize[elemwise_vectorize, simd_width](len(self))
        return new_mat

    @always_inline
    fn _elemwise_tensor_tensor[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, mat: Self) -> Self:
        alias simd_width: Int = self.simd_width
        var new_mat = Self(len(self))
        @parameter
        fn tensor_tensor_vectorize[simd_width: Int](idx: Int) -> None:
            new_mat._matPtr.store[width=simd_width](idx, func[dtype, simd_width](self._matPtr.load[width=simd_width](idx), mat._matPtr.load[width=simd_width](idx)))
        vectorize[tensor_tensor_vectorize, simd_width](len(self))
        return new_mat

    @always_inline
    fn _elemwise_pow(self, p: Int) -> Self:
        alias simd_width = self.simd_width
        var new_mat = Self(self.rows,self.cols)
        @parameter
        fn pow_vectorize[simd_width: Int](idx: Int) -> None:
            new_mat._matPtr.store[width=simd_width](idx, math.pow(self._matPtr.load[width=simd_width](idx), p))
        vectorize[pow_vectorize, simd_width](len(self))
        return new_mat