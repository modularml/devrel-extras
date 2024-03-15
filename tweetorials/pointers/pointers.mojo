from testing import assert_equal, assert_not_equal, assert_false


@value
@register_passable("trivial")
struct MyType(Stringable):
    var x: Int
    var y: SIMD[DType.float32, 4]

    fn __str__(self) -> String:
        return "MyType { " + str(self.x) + ", " + str(self.y) + " }"


struct MoveIt(Movable, Stringable):
    var s: String

    fn __init__(inout self, s: String):
        self.s = "MoveIt! " + s

    fn __moveinit__(inout self, owned existing: Self):
        self.s = existing.s ^

    fn __str__(self) -> String:
        return self.s


def main():
    # Unsafe Pointer
    print("Create null pointer")
    ptr = Pointer[Int]()
    # # or via
    null_ptr = Pointer[Int]().get_null()
    assert_equal(ptr, null_ptr, "ptr must be null")
    print("ptr:", ptr)  # 0x0
    print("int(ptr):", int(ptr))  # 0
    assert_equal(hex(int(ptr)), ptr, "hex representation must match")

    ptr = ptr.alloc(2)  # heap allocates for 2 Ints and returns a new pointer
    ptr.store(42)
    assert_not_equal(ptr, null_ptr, "ptr is not null")
    print("ptr.load():", ptr.load())  # 42
    ptr2 = ptr.offset(1)  # returns a new pointer ptr2 shifted by 1
    ptr2.store(-42)
    print("ptr2.load():", ptr2.load())
    print("ptr.free() deallocates Int from heap but it doesn't become a null pointer")
    ptr.free()
    ptr2.free()
    assert_not_equal(ptr, null_ptr, "ptr is not null")
    print(
        "ptr.load() loads some random value each time the program is run:", ptr.load()
    )

    my_ptr = Pointer[MyType]()
    my_ptr = my_ptr.alloc(1)
    y = SIMD[DType.float32, 4]()
    for i in range(4):
        y[i] = i

    my_ptr.store(MyType(42, y))
    print("my_ptr.load():", my_ptr.load())  # MyType { 42, [0.0, 1.0, 2.0, 3.0] }
    my_ptr.free()

    # AnyPointer
    anyptr = AnyPointer[MoveIt]()
    assert_equal(anyptr, null_ptr)
    print("empty anyptr:", anyptr)
    anyptr = anyptr.alloc(1)
    anyptr.emplace_value(MoveIt("MoveIt!"))
    assert_not_equal(anyptr, null_ptr)
    print("anyptr:", anyptr)
    print("dereference via anyptr[]:", anyptr[])  # MoveIt! MoveIt!
    taken = anyptr.take_value()  # takes the value out
    print("anyptr.take_value():", taken)  # MoveIt! MoveIt!
    # NOTE: anyptr value is moved and dereferencing is invalid use-after-free UB
    print("Undefined Behaviour: anyptr[] is invalid use-after-tree deref")

    print("Convert Pointer to AnyPointer with `__from_index` method")
    new_anyptr = AnyPointer[MyType].__from_index(int(my_ptr))
    print(new_anyptr[])  # MyType { 42, [0.0, 1.0, 2.0, 3.0] }
    new_anyptr.free()
    back_to_ptr = Pointer[MyType].__from_index(int(new_anyptr))
    print(back_to_ptr.load())  # MyType { 42, [0.0, 1.0, 2.0, 3.0] }
    back_to_ptr.free()
