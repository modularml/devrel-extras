from math import mod, trunc, align_down, align_down_residual
from memory import memset_zero, memcpy
from sys.info import simdwidthof
from algorithm import vectorize
from algorithm.functional import elementwise
from random import rand
from sys.intrinsics import strided_load
from python import Python
import benchmark
from benchmark import Unit
from testing import assert_true, assert_equal

struct MojoMatrix[dtype: DType = DType.float64, is_row_major: Bool=True]():
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
        self._matPtr = other._matPtr
        self.rows = other.rows
        self.cols = other.cols

    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[dtype,1]:
        return self._matPtr.load(idx) 

    @always_inline
    fn __getitem__(self, x: Int, y: Int) -> SIMD[dtype,1]:
        return self._matPtr.load(x * self.cols + y)

    @always_inline
    fn __len__(self) -> Int:
        return self.rows*self.cols

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

    @always_inline
    fn to_colmajor(self) raises -> MojoMatrix[dtype,False]:
        assert_true(self.is_row_major == True, "Matrix must be in row-major format, to convert to column-major.")
        return MojoMatrix[dtype, False](self.rows, self.cols, self._transpose_order())

    @always_inline
    fn transpose(self) -> MojoMatrix[dtype, is_row_major]:
        return MojoMatrix[dtype, is_row_major](self.cols, self.rows, self._transpose_order())

    @always_inline
    fn mean(self, owned cols_list:List[Int]) -> Self:
        var new_mat = Self(1,len(cols_list))
        alias simd_width: Int = self.simd_width
        var simd_sum = SIMD[dtype, simd_width](0)
        var simd_multiple = align_down(self.rows, simd_width)
        for idx_col in range(len(cols_list)):
            simd_sum = simd_sum.splat(0)
            if is_row_major:
                var tmpPtr = self._matPtr.offset(cols_list[idx_col])
                for idx_row in range(0, simd_multiple, simd_width):
                    simd_sum+=tmpPtr.simd_strided_load[width=simd_width](self.cols)
                    tmpPtr = tmpPtr.offset(simd_width*self.cols)
                for idx_row in range(simd_multiple, self.rows):
                    simd_sum[0] += tmpPtr.simd_strided_load[width=1](self.cols)
                    tmpPtr = tmpPtr.offset(simd_width*self.cols)
                new_mat._matPtr[idx_col] = simd_sum.reduce_add()
            else:
                for idx_row in range(0, simd_multiple, simd_width):
                    simd_sum+=self._matPtr.load[width=simd_width](self.rows*cols_list[idx_col]+idx_row)
                for idx_row in range(simd_multiple, self.rows):
                    simd_sum[0] += self._matPtr.load[width=1](self.rows*cols_list[idx_col]+idx_row)
                new_mat._matPtr[idx_col] = simd_sum.reduce_add()
            new_mat._matPtr[idx_col] /= self.rows
        return new_mat

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
        return Self(rows,cols,_matPtr)

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
        return np_arr

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
                    if is_row_major:
                        val = self[i,j]
                    else:
                        val = self._matPtr[i+j*rows]
                var int_str: String
                if val >= 0.0:
                    int_str = str(trunc(val).cast[DType.index]())
                else:
                    int_str = "-"+str(trunc(val).cast[DType.index]())
                    val = -val
                var float_str: String = ""
                if mod(val,1)==0:
                    float_str = "0"
                else:
                    try:
                        float_str = str(mod(val,1)).split('.')[-1]
                    except:
                        return ""
                var s: String = int_str+"."+float_str
                if j==0:
                    printStr+=s
                else:
                    printStr+="  "+s
            printStr+="]"
        if rank>1:
            printStr+="]"
        printStr+="\n"
        if rank>2:
            printStr+="]"
        var row_col_str: String
        if is_row_major:
            row_col_str = "Row Major"
        else:
            row_col_str = "Column Major"
        printStr+="Matrix: "+str(self.rows)+'x'+str(self.cols)+" | "+"DType:"+str(dtype)+" | "+row_col_str+"\n"
        return printStr
