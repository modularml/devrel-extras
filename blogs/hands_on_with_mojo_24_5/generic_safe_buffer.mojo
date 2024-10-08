from collections import Optional


trait StringableFormattableCollectionElement(Formattable, StringableCollectionElement):
    ...


struct SafeBuffer[T: CollectionElement]:
    var _data: UnsafePointer[Optional[T]]
    var size: Int

    fn __init__(inout self, size: Int):
        debug_assert(size > 0, "size must be greater than zero")
        self._data = UnsafePointer[Optional[T]].alloc(size)
        for i in range(size):
            (self._data + i).init_pointee_copy(NoneType())

        self.size = size

    @staticmethod
    fn initialize_with_value(size: Int, value: T) -> Self as output:
        output = SafeBuffer[T](size)
        for i in range(size):
            output.write(i, value)

        return

    fn __copyinit__(inout self, existing: Self):
        self._data = existing._data
        self.size = existing.size

    fn __moveinit__(inout self, owned existing: Self):
        self._data = existing._data
        self.size = existing.size

    fn __del__(owned self):
        self._data.free()

    fn write(inout self, index: Int, value: Optional[T]):
        debug_assert(0 <= index < self.size, "index must be within the buffer")
        self._data[index] = value

    fn read(self, index: Int) -> Optional[T]:
        debug_assert(0 <= index < self.size, "index must be within the buffer")
        return self._data[index]

    fn __str__[U: StringableFormattableCollectionElement](self: SafeBuffer[U]) -> String:
        ret = String()
        writer = ret._unsafe_to_formatter()
        self.format_to(writer)
        _ = writer^
        return ret^

    fn format_to[
        U: StringableFormattableCollectionElement
    ](self: SafeBuffer[U], inout writer: Formatter):
        debug_assert(self.size > 0, "size must be greater than zero")
        writer.write("[")
        for i in range(self.size - 1):
            if self._data[i]:
                writer.write(self._data[i].value(), ", ")
            else:
                writer.write("None", ", ")

        if self._data[self.size - 1]:
            writer.write(self._data[self.size - 1].value())
        else:
            writer.write("None")

        writer.write("]")

    fn take(inout self, index: Int) -> Optional[T] as output:
        output = self.read(index)
        self.write(index, Optional[T](None))
        return


fn process_buffers[T: CollectionElement](buffer1: SafeBuffer[T], inout buffer2: SafeBuffer[T]):
    debug_assert(buffer1.size == buffer2.size, "buffer sizes much match")
    for i in range(buffer1.size):
        buffer2.write(i, buffer1.read(i))


struct NotStringableNorFormattable(CollectionElement):
    fn __init__(inout self):
        ...

    fn __copyinit__(inout self, existing: Self):
        ...

    fn __moveinit__(inout self, owned existing: Self):
        ...


def main():
    buffer1 = SafeBuffer[UInt8].initialize_with_value(size=10, value=UInt8(128))
    buffer2 = SafeBuffer[UInt8](size=10)
    # process_buffers(buffer1, buffer1) # <-- argument exclusivity detects such errors at compile time
    process_buffers(buffer1, buffer2)
    # testing conditional conformance
    print(buffer2.__str__())
    print(buffer2.take(0).value())
    print(buffer2.__str__())

    sbuffer1 = SafeBuffer[String].initialize_with_value(size=10, value=String("hi"))
    print(sbuffer1.take(5).value())
    print(sbuffer1.__str__())

    ## uncomment to see the compiler error:
    # buf = SafeBuffer[NotStringableNorFormattable](10)
    # buf.__str__()

    """
    Compiler error message:
    /Users/ehsan/workspace/devrel-extras/blogs/hands_on_with_mojo_24_5/generic_safe_buffer.mojo:106:16: error: invalid call to '__str__': could not deduce parameter 'U' of callee '__str__'
        buf.__str__()
        ~~~~~~~~~~~^~
    /Users/ehsan/workspace/devrel-extras/blogs/hands_on_with_mojo_24_5/generic_safe_buffer.mojo:106:5: note: failed to infer parameter 'U', argument type 'NotStringableNorFormattable' does not conform to trait 'StringableFormattableCollectionElement'
        buf.__str__()
        ^~~
    /Users/ehsan/workspace/devrel-extras/blogs/hands_on_with_mojo_24_5/generic_safe_buffer.mojo:48:8: note: function declared here
        fn __str__[U: StringableFormattableCollectionElement](self: SafeBuffer[U]) -> String:
        ^
    mojo: error: failed to parse the provided Mojo source module
    """
