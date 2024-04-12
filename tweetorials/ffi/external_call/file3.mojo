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

def fclose(stream: Pointer[FILE]):
    debug_assert(stream != Pointer[FILE](), "File must be opened first")
    ret = _fclose(stream)
    if ret:
        raise Error("File cannot be closed")

    return

fn _fclose(stream: Pointer[FILE]) -> c_int:
    return external_call["fclose", c_int, Pointer[FILE]](stream)

def fopen(path: String, mode: String) -> Pointer[FILE]:
    path_ptr = _as_char_ptr(path)
    mode_ptr = _as_char_ptr(mode)
    stream = _fopen(path_ptr, mode_ptr)
    if _ferror(stream):
        raise Error("Cannot open the file")

    mode_ptr.free()
    path_ptr.free()
    return stream

fn _fopen(path: Pointer[c_char], mode: Pointer[c_char]) -> Pointer[FILE]:
    return external_call[
        "fopen",
        Pointer[FILE],
        Pointer[c_char],
        Pointer[c_char],
    ](path, mode)

fn _ferror(stream: Pointer[FILE]) -> c_int:
    return external_call["ferror", c_int, Pointer[FILE]](stream)

fn _as_char_ptr(s: String) -> Pointer[c_char]:
    var nelem = len(s)
    var ptr = Pointer[c_char]().alloc(nelem + 1)  # +1 for null termination
    for i in range(len(s)):
        ptr.store(i, ord(s[i]))

    ptr.store(nelem, 0)  # null-terminate the string
    return ptr

fn _fseek(stream: Pointer[FILE], offset: c_long, whence: c_int) -> c_int:
    return external_call["fseek", c_int, Pointer[FILE], c_long, c_int](
        stream, offset, whence
    )

def fseek(stream: Pointer[FILE], offset: UInt64 = 0, whence: Int32 = SEEK_END):
    debug_assert(stream != Pointer[FILE](), "File must be opened first")
    ret = _fseek(stream, offset, whence)
    if ret:
        fclose(stream)
        raise Error("Error seeking in file")

    return

fn ftell(stream: Pointer[FILE]) -> c_long:
    debug_assert(stream != Pointer[FILE](), "File must be opened")
    return external_call["ftell", c_long, Pointer[FILE]](stream)

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

def fread(stream: Pointer[FILE], buf_read_size: Int = 1024) -> String:
    debug_assert(stream != Pointer[FILE](), "File must be opened first")
    # Choosing a large buffer for the sake of example.
    # Exercise: Implement `read_file_to_end` in case
    # the size is greater than the buffer size
    buf = Pointer[c_char]().alloc(buf_read_size + 1)  # +1 for null termination
    count = _fread(buf.bitcast[c_void](), 1, buf_read_size, stream)
    if count <= 0:
        if count < 0:
            fclose(stream)
            raise Error("Cannot read data")
        else:
            print("End of file reached")

    buf.store(count, 0)  # null-terminate
    # String owns the ptr so not need to call `free`
    # String len is Int and count is Int32
    return String(buf.bitcast[Int8](), int(count) + 1) # +1 include null-termintor


def main():
    with assert_raises():
        null_ptr = Pointer[FILE]()
        fclose(null_ptr)
        _ = ftell(null_ptr)
        fseek(null_ptr)
        _ = fread(null_ptr)
    try:
        fp = fopen("test.txt", "r")
        fseek(fp)
        size = ftell(fp)
        print("file size in bytes:", size) # 36 bytes
        # Reposition to the start of the file
        fseek(fp, whence=SEEK_SET)
        print(fread(fp))
        _ = fclose(fp)
    except:
        print("Error has occured")
