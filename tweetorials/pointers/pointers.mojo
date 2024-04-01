from testing import assert_equal, assert_not_equal, assert_false
from sys.info import alignof

@value
@register_passable("trivial")
struct MyType(Stringable):
    var x: Int
    var y: SIMD[DType.float32, 4]

    fn __str__(self) -> String:
        return "MyType { " + str(self.x) + ", " + str(self.y) + " }"

def main():
    # Unsafe Pointer
    print("Create null pointer")
    ptr = Pointer[Int]()
    # or via
    null_ptr = Pointer[Int]().get_null()
    assert_equal(ptr, null_ptr, "ptr must be null")
    print("ptr:", ptr)  # 0x0
    print("int(ptr):", int(ptr))  # 0
    assert_equal(hex(int(ptr)), ptr, "hex representations must match")

    print("Size of Int in bytes:", sizeof[Int]()) # 8 bytes
    print("Int alignment in bytes:", alignof[Int]()) # 8 bytes
    print("Size of Pointer[Int] in bytes:", sizeof[Pointer[Int]]()) # 8 bytes on x86-64
    print("Pointer[Int] alignment in bytes:", alignof[Pointer[Int]]()) # 8 bytes
    ptr = ptr.alloc(2)  # heap allocates for 2 Ints (with 8 bytes alignment) and returns a new pointer
    ptr.store(42)
    assert_not_equal(ptr, null_ptr, "ptr is not null")
    print("ptr.load():", ptr.load())  # 42
    ptr2 = ptr.offset(1)  # returns a new pointer ptr2 shifted by 1
    ptr2.store(-42)
    print("ptr2.load():", ptr2.load()) # -42

    print("ptr.free() deallocates Int from heap but it doesn't become a null pointer")
    ptr.free()
    print("Note ptr2 has also been freed")
    assert_not_equal(ptr, null_ptr, "ptr is not null")
    print(
        "Be careful: ptr.load() is UB and may load some random value each time the program is run:", ptr.load()
    )

    my_ptr = Pointer[MyType]()
    print("Size of MyType in bytes:", sizeof[MyType]()) # 32 bytes
    print("Size of Pointer[MyType] in bytes:", sizeof[Pointer[MyType]]()) # 8 bytes
    print("MyType alignment in bytes:", alignof[MyType]()) # 16 bytes
    print("Pointer[MyType] alignment in bytes:", alignof[Pointer[MyType]]()) # 8 bytes
    # let's use a different alignment than the default 32 which is the size of `MyType`
    my_ptr = my_ptr.alloc(1, alignment=31)
    # in above we have allocated 62 bytes to ensure alignment.
    y = SIMD[DType.float32, 4]()
    for i in range(4):
        y[i] = i

    my_ptr.store(MyType(42, y))
    print("my_ptr.load():", my_ptr.load())  # MyType { 42, [0.0, 1.0, 2.0, 3.0] }
    my_ptr.free()

    alias dtype = DType.int8
    alias simd_width = simdwidthof[dtype]()
    print("SIMD width for DType.int8:", simd_width) # 64 on my machine
    my_dptr = DTypePointer[dtype].alloc(simd_width)
    # fills with zero
    memset_zero(my_dptr, simd_width)
    print("Load a single element via .load():", my_dptr.load()) # 0
    print("Load a simd vector via .load() with offset 0:", my_dptr.load[width=simd_width](0)) # [0, ..., 0]

    vec = SIMD[dtype, simd_width]()
    for i in range(simd_width):
        vec[i] = i

    my_dptr.store[width=simd_width](0, vec)
    print(my_dptr.load[width=simd_width]()) # [0, 1, ..., 63]
    my_dptr.free()
