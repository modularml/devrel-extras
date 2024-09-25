struct UnsafeBuffer:
    var data: UnsafePointer[UInt8]
    var size: Int

    fn __init__(inout self, size: Int):
        self.data = UnsafePointer[UInt8].alloc(size)
        self.size = size

    fn write(inout self, index: Int, value: UInt8):
        # note `self.data` is uninitialized so we have to use `init_pointee_copy/move`
        # methods to safely initialize the allocated memory
        self.data.init_pointee_copy(value)

    fn read(self, index: Int) -> UInt8:
        return self.data[index]

    fn __del__(owned self):
        self.data.free()


fn main():
    ub = UnsafeBuffer(10)
    ub.write(0, 255)
    ub.write(1, 128)
    print("unsafe buffer outputs:")
    print(ub.read(0))
    print("the data of the unsafe buffer is freed here bc there's no lifetime associate with it")
    print(ub.read(1))
