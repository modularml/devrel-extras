from sys.ffi import external_call

alias c_char = UInt8

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

alias c_int = Int32

fn fclose(stream: Pointer[FILE]) -> c_int:
    return external_call["fclose", c_int, Pointer[FILE]](stream)

fn ferror(stream: Pointer[FILE]) -> c_int:
    return external_call["ferror", c_int, Pointer[FILE]](stream)

fn to_char_ptr(s: String) -> Pointer[c_char]:
    var nelem = len(s)
    var ptr = Pointer[c_char]().alloc(nelem + 1)  # +1 for null termination
    for i in range(len(s)):
        ptr.store(i, ord(s[i]))

    ptr.store(nelem, 0)  # null-terminate the string
    return ptr

def main():
    fp = fopen(to_char_ptr("test.txt"), to_char_ptr("r"))
    if ferror(fp):
        print("Error opening file")
        return

    _ = fclose(fp)
