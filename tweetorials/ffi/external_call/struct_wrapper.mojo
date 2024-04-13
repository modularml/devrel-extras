from sys.ffi import external_call
from testing import assert_raises

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
        # https://man7.org/linux/man-pages/man3/fopen.3.html
        var handle = external_call["fopen", Pointer[FILE]](
            path_ptr, mode_ptr
        )
        mode_ptr.free()
        path_ptr.free()
        if handle == Pointer[FILE]():
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
        """Safe and idiomatic wrapper https://man7.org/linux/man-pages/man3/fclose.3.html."""
        debug_assert(self.handle != Pointer[FILE](), "File must be opened first")
        var ret = external_call["fclose", c_int, Pointer[FILE]](self.handle)
        if ret:
            raise Error("Error in closing the file")

        return

    fn fseek(self, offset: UInt64 = 0, whence: Int32 = SEEK_END) raises:
        """Safe and idiomatic wrapper https://man7.org/linux/man-pages/man3/fseek.3.html."""
        debug_assert(self.handle != Pointer[FILE](), "File must be opened first")
        var ret = external_call["fseek", c_int, Pointer[FILE], c_long, c_int](
            self.handle, offset, whence
        )
        if ret:
            self.fclose()
            raise Error("Error seeking in file")

        return

    fn ftell(self) raises -> UInt64:
        """Safe and idiomatic wrapper https://man7.org/linux/man-pages/man3/ftell.3p.html."""
        debug_assert(self.handle != Pointer[FILE](), "File must be opened")
        var ret = external_call["ftell", c_long, Pointer[FILE]](self.handle)
        if ret == -1:
            raise Error("ftell failed")

        return ret

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
        """Safe and idiomatic wrapper https://man7.org/linux/man-pages/man3/fread.3p.html."""
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



fn fopen(path: String, mode: String = "r") raises -> FileHandle:
    return FileHandle(path, mode)

def main():
    try:
        file = fopen("test.txt")
        file.fseek(0)
        size = file.ftell()
        print("file size in bytes:", size) # 36 bytes
        file.fseek(whence=SEEK_SET)
        print(file.fread())
        file.fclose()
    except:
        print("Error has occured")

    with assert_raises():
        # test double close
        with assert_raises():
            file = fopen("test.txt")
            file.fclose()
            file.fclose()
        # test notexist
        _ = fopen("notexist.txt")
        # test fseek and ftell fail cases
        file = fopen("test.txt")
        file.fseek(-100)
        _ = file.ftell()
        file.fclose()
