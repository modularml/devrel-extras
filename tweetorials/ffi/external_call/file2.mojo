from sys.ffi import external_call

alias c_char = UInt8
alias c_int = Int32

@register_passable("trivial")
struct FILE:
    ...

fn fopen(path: Pointer[c_char], mode: Pointer[c_char]) -> Pointer[FILE]:
    return external_call[
        "fopen",
        Pointer[FILE],
        Pointer[c_char],
        Pointer[c_char],
    ](path, mode)

fn ferror(stream: Pointer[FILE]) -> c_int:
    return external_call["ferror", c_int, Pointer[FILE]](stream)

fn fclose(stream: Pointer[FILE]) -> c_int:
    return external_call["fclose", c_int, Pointer[FILE]](stream)

fn as_char_ptr(s: String) -> Pointer[c_char]:
    var nelem = len(s)
    var ptr = Pointer[c_char]().alloc(nelem + 1)  # +1 for null termination
    for i in range(len(s)):
        ptr.store(i, ord(s[i]))

    ptr.store(nelem, 0)  # null-terminate the string
    return ptr


alias c_long = UInt64

fn fseek(stream: Pointer[FILE], offset: c_long, whence: c_int) -> c_int:
    return external_call["fseek", c_int, Pointer[FILE], c_long, c_int](
        stream, offset, whence
    )

fn ftell(stream: Pointer[FILE]) -> c_long:
    return external_call["ftell", c_long, Pointer[FILE]](stream)

alias c_void = UInt8
alias c_size_t = Int

fn fread(
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

def main():
    path_ptr = as_char_ptr("test.txt")
    mode_ptr = as_char_ptr("r")
    fp = fopen(path_ptr, mode_ptr)
    if ferror(fp):
        print("Error opening file")
        return

    alias SEEK_END = 2
    if fseek(fp, 0, SEEK_END):
        print("Error seeking in file")
        _ = fclose(fp)
        return

    size = ftell(fp)
    print("file size in bytes:", size) # 36 bytes
    # Reposition to the start of the file
    alias SEEK_SET = 0
    if fseek(fp, 0, SEEK_SET):
        print("Error repositioning to the start of the file")
        _ = fclose(fp)
        return

    # Choosing a large buffer for the sake of example.
    # Exercise: Implement `read_file_to_end` in case
    # the size is greater than the buffer size
    buf_read_size = 1024
    buf = Pointer[c_char]().alloc(buf_read_size + 1)  # +1 for null termination
    count = fread(buf.bitcast[c_void](), 1, buf_read_size, fp)
    if count <= 0:
        if count < 0:
            _ = fclose(fp)
            print("Cannot read data")
        else:
            print("End of file reached")

    buf.store(count, 0)  # null-terminate
    # String owns the ptr so not need to call `free`
    # String len is Int and count is Int32
    print(String(buf.bitcast[Int8](), int(count) + 1)) # +1 include null-termintor
    _ = fclose(fp)
    mode_ptr.free()
    path_ptr.free()
