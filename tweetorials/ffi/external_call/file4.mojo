from sys.ffi import external_call

alias c_char = UInt8
alias c_int = Int32
alias c_long = UInt64
alias c_void = UInt8
alias c_size_t = Int

alias SEEK_SET = 0
alias SEEK_END = 2

@register_passable("trivial")
struct FILE:
    ...

struct FileHandle:
    var handle: Pointer[FILE]

    fn __init__(inout self, path: String, mode: String) raises:
        var path_ptr = self._as_char_ptr(path)
        var mode_ptr = self._as_char_ptr(mode)
        var handle = external_call["fopen", Pointer[FILE]](
            path_ptr, mode_ptr
        )
        mode_ptr.free()
        path_ptr.free()
        var err = external_call["ferror", c_int, Pointer[FILE]](handle)
        if err:
            raise Error("Error opening file")

        self.handle = handle

    @staticmethod
    fn _as_char_ptr(s: String) -> Pointer[c_char]:
        var nelem = len(s)
        var ptr = Pointer[c_char]().alloc(nelem + 1)  # +1 for null termination
        for i in range(len(s)):
            ptr.store(i, ord(s[i]))

        ptr.store(nelem, 0)  # null-terminate the string
        return ptr


    fn fclose(self) raises:
        debug_assert(self.handle != Pointer[FILE](), "File must be opened first")
        var ret = external_call["fclose", c_int, Pointer[FILE]](self.handle)
        if ret:
            raise Error("Error in closing the file")

        return

    fn fseek(self, offset: UInt64 = 0, whence: Int32 = SEEK_END) raises:
        debug_assert(self.handle != Pointer[FILE](), "File must be opened first")
        var ret = external_call["fseek", c_int, Pointer[FILE], c_long, c_int](
            self.handle, offset, whence
        )
        if ret:
            self.fclose()
            raise Error("Error seeking in file")

        return

    fn ftell(self) -> UInt64:
        debug_assert(self.handle != Pointer[FILE](), "File must be opened")
        return external_call["ftell", c_long, Pointer[FILE]](self.handle)

    @staticmethod
    fn _fread(
        ptr: Pointer[c_void],
        size: c_size_t,
        nitems: c_size_t,
        stream: Pointer[FILE],
    ) -> c_int:
        return external_call[
            "fread",
            c_size_t,
            Pointer[c_void],
            c_size_t,
            c_size_t,
            Pointer[FILE],
        ](ptr, size, nitems, stream)

    fn fread(self, buf_read_size: Int = 1024) raises -> String:
        debug_assert(self.handle != Pointer[FILE](), "File must be opened first")
        # Choosing a large buffer for the sake of example.
        # Exercise: Implement `read_file_to_end` in case
        # the size is greater than the buffer size
        var buf = Pointer[c_char]().alloc(buf_read_size + 1)  # +1 for null termination
        var count = self._fread(buf.bitcast[c_void](), 1, buf_read_size, self.handle)
        if count <= 0:
            if count < 0:
                self.fclose()
                raise Error("Cannot read data")
            else:
                print("End of file reached")

        buf.store(count, 0)  # null-terminate
        # String owns the ptr so not need to call `free`
        # String len is Int and count is Int32
        return String(buf.bitcast[Int8](), int(count) + 1) # +1 include null-termintor


fn fopen(path: String, mode: String) raises -> FileHandle:
    return FileHandle(path, mode)

def main():
    try:
        file = fopen("test.txt", "r")
        file.fseek(0)
        size = file.ftell()
        print("file size in bytes:", size) # 36 bytes
        file.fseek(whence=SEEK_SET)
        print(file.fread())
        file.fclose()
    except:
        print("Error has occured")
