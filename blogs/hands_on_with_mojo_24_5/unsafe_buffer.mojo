from memory import memset_zero


struct UnsafeBuffer:
    var data: UnsafePointer[UInt8]
    var size: Int

    fn __init__(inout self, size: Int):
        self.data = UnsafePointer[UInt8].alloc(size)
        memset_zero(self.data, size)
        self.size = size

    fn __del__(owned self):
        self.data.free()


def main():
    ub = UnsafeBuffer(10)
    print("initial value at index 0:")
    print(ub.data[0])
    ub.data[0] = 255
    print("value at index 0 after getting set to 255:")
    print(ub.data[0])
