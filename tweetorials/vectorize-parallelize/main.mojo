def main():
    alias size = 10
    alias type = DType.int32
    alias simd_width = simdwidthof[type]()
    # on my machine it is 4 i.e. 4 x int32 which is 128 SIMD register
    print("simd_width:", simd_width)
    x = DTypePointer[type].alloc(size)
    for i in range(size):
        x[i] = 42

    # Note: x.load returns SIMD[type, width]
    print("initialized x:", x.load[width=size]())  # [42, ..., 42]
    print("manual SIMD print")
    simd_multiple = size - size % simd_width  # 10 - 10 % 4 = 8
    for offset in range(0, simd_multiple, simd_width):
        print(
            "offset =",
            offset,
            "width =",
            simd_width,
            ":",
            x.load[width=simd_width](offset),
        )

    # outputs:
    # offset = 0 width = 4 : [42, 42, 42, 42]
    # offset = 4 width = 4 : [42, 42, 42, 42]

    for offset in range(simd_multiple, size):
        print(
            "offset =",
            offset,
            "width = 1",
            ":",
            x.load[width=1](),
        )

    # outputs:
    # offset = 8 width = 1 : 42
    # offset = 9 width = 1 : 42
    # offset = 10 width = 1 : 42

    print("vectorized SIMD print")
    from algorithm import vectorize

    # Note: Type of `offset` is `Int` which is used for indexing
    @parameter
    fn print_it[width: Int](offset: Int):
        print(
            "offset =",
            offset,
            "width =",
            width,
            ":",
            x.load[width=width](offset=offset),
        )

    vectorize[print_it, simd_width](size)
    # outputs:
    # offset = 0 width =  4 : [42, 42, 42, 42]
    # offset = 4 width =  4 : [42, 42, 42, 42]
    # offset = 8 width =  1 : 42
    # offset = 9 width =  1 : 42

    # Notice how much `vectorize` simplifies and
    # takes care of the remainder
    # `range(simd_multiple, size)` above and adjusts `width`

    y = DTypePointer[type].alloc(size)
    for j in range(size):
        y[j] = -42

    print("initialized y:", y.load[width=size]())  # [-42, ..., -42]
    z = DTypePointer[type].alloc(size)
    # initialize with 0
    memset_zero(z, size)

    @parameter
    fn elementwise_sum[width: Int](offset: Int):
        # elementwise sum formula is:
        var x_simd_chunk = x.load[width=width](offset=offset)
        print(
            "[x]    ",
            "offset =",
            offset,
            "width = ",
            width,
            ":",
            x_simd_chunk,
        )
        var y_simd_chunk = y.load[width=width](offset=offset)
        print(
            "[y]    ",
            "offset =",
            offset,
            "width = ",
            width,
            ":",
            y_simd_chunk,
        )
        var val = x_simd_chunk + y_simd_chunk  # SIMD sum
        print(
            "[x + y]",
            "offset =",
            offset,
            "width = ",
            width,
            ":",
            val,
        )
        print("==============================================================")
        z.store[width=width](offset=offset, val=val)

    vectorize[elementwise_sum, simd_width](size)
    # outputs:
    # [x]     offset = 0 width =  4 : [42, 42, 42, 42]
    # [y]     offset = 0 width =  4 : [-42, -42, -42, -42]
    # [x + y] offset = 0 width =  4 : [0, 0, 0, 0]
    # ==============================================================
    # [x]     offset = 4 width =  4 : [42, 42, 42, 42]
    # [y]     offset = 4 width =  4 : [-42, -42, -42, -42]
    # [x + y] offset = 4 width =  4 : [0, 0, 0, 0]
    # ==============================================================
    # [x]     offset = 8 width =  1 : 42
    # [y]     offset = 8 width =  1 : -42
    # [x + y] offset = 8 width =  1 : 0
    # ==============================================================
    # [x]     offset = 9 width =  1 : 42
    # [y]     offset = 9 width =  1 : -42
    # [x + y] offset = 9 width =  1 : 0
    # ==============================================================
    print("elementwise sum result:", z.load[width=size]())  # [0, ..., 0]

    # don't forget to free the allocated pointers
    z.free()
    y.free()
    x.free()

    import math
    from algorithm import parallelize

    alias large_size = size * size  # 100
    xx = DTypePointer[type].alloc(large_size)
    # xx is [0, 1, ..., 99]
    for i in range(large_size):
        xx[i] = i

    yy = DTypePointer[type].alloc(large_size)
    # yy is [10, 10, ..., 10]
    for i in range(large_size):
        yy[i] = 10

    zz = DTypePointer[type].alloc(large_size)
    # initialize with 0
    memset_zero(zz, large_size)

    num_work_items = 2
    print("num_work_items =", num_work_items)
    chunk_size = math.div_ceil(large_size, num_work_items)  # 50
    print("chunk_size =", chunk_size)

    # divides xx, yy into two arrayes of 50 elements each (xx_1, xx_2) and (yy_1, yy_2)
    # and performs the vectorized elementwise sum using 2 threads in parallel
    # i.e. (xx_1 + yy_1) and (xx_2 + yy_2) in parallel
    # +-----------------------------------------+
    # |           Parallel Elementwise Sum      |
    # +-----------------------------------------+
    # |                                         |
    # |          +----------------+             |
    # |          |     Thread 1   |             |
    # |          |                |             |
    # |  Array 1 | xx_1 + yy_1 -> zz_1          |
    # |          |  [0:50]        |             |
    # |          +----------------+             |
    # |                                         |
    # |          +----------------+             |
    # |          |     Thread 2   |             |
    # |          |                |             |
    # |  Array 2 | xx_2 + yy_2 -> zz_2          |
    # |          |  [50:100]      |             |
    # |          +----------------+             |
    # |                                         |
    # +-----------------------------------------+
    # | Result: zz = [zz_1, zz_2]               |
    # |         [10, 11, ..., 109]              |
    # +-----------------------------------------+

    @parameter
    fn parallel_elementwise_sum(thread_id: Int):
        # NOTE: for parallelize do not include any
        # I/O such as print which is not thread-safe
        var start = thread_id * chunk_size
        var end = math.min(start + chunk_size, large_size)

        @parameter
        fn _elementwise_sum[width: Int](offset: Int):
            var val = xx.load[width=width](offset=start + offset) + yy.load[width=width](offset=start + offset)
            zz.store[width=width](offset=start + offset, val=val)

        vectorize[_elementwise_sum, simd_width](end - start)

    parallelize[parallel_elementwise_sum](num_work_items)
    print("parallel elementwise sum result:", zz.load[width=large_size]())
    # output:
    # [10, 11, ..., 109]

    # don't forget to free the pointers
    zz.free()
    yy.free()
    xx.free()
