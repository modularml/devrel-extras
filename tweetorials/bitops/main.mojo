from math.bit import ctlz, cttz, bit_and, bit_not
from math import reduce_bit_count, rotate_bits_left
from testing import assert_equal


def main():
    print("shift left 1 << 2:", 1 << 2, "shift right 3 >> 1:", 3 >> 1)  # 4 and 1, resp.
    print(
        "rotate bits left/right in usigned SIMD",
        rotate_bits_left[shift=1](SIMD[DType.uint8, 2](1, 2)),
    )  # [2, 4]

    print("count leading zeros in binary '00000001'", ctlz(Int8(1)))  # 7
    print("count trailing zeros in binary '00000001'", cttz(Int8(1)))  # 0
    # `ctlz` and `cttz` work with integer types
    assert_equal(ctlz(Int32(1)), ctlz(UInt32(1)))

    # also SIMD vectors
    print("count leading zeros:", ctlz(SIMD[DType.int8, 4](0, 1, 2, 3)))  # [8, 7, 6, 6]
    print("count leading zeros:", cttz(SIMD[DType.int8, 4](0, 1, 2, 3)))  # [8, 0, 1, 0]

    # Check for power of 2 using bit counts
    def is_power_of_two_count(x: SIMD) -> SIMD[DType.bool, x.size]:
        return ctlz(x) + cttz(x) == x.type.bitwidth() - 1

    # More efficient way to check for power of 2 `x & (x - 1) == 0`
    # while checking for sign
    def is_power_of_two(x: SIMD) -> SIMD[DType.bool, x.size]:
        # x & (x - 1) is the same as bit_and(x, x - 1)
        # Note: x != 0 checks for sign
        return (x & (x - 1) == 0) & (x != 0)

    assert_equal(
        is_power_of_two(SIMD[DType.int8, 4](-1, 0, 1, 2)),
        SIMD[DType.bool, 4](False, False, True, True),
    )
    assert_equal(
        is_power_of_two_count(SIMD[DType.int8, 4](-1, 0, 1, 2)),
        SIMD[DType.bool, 4](False, False, True, True),
    )

    # Now given only `bit_and` i.e. `&` and `bit_not` i.e. `~`,
    # we can implement `bit_or`, `bit_xor` and `bit_xnor`
    def bit_or[
        type: DType, size: Int
    ](a: SIMD[type, size], b: SIMD[type, size]) -> SIMD[type, size]:
        # De Morgan's law: A ∨ B = ¬(¬A ∧ ¬B)
        # the same `bit_not(bit_and(bit_not(a), bit_not(b)))`
        return ~(~a & ~b)

    def bit_xor[
        type: DType, size: Int
    ](a: SIMD[type, size], b: SIMD[type, size]) -> SIMD[type, size]:
        # Exclusive OR: A ⊕ B = (A ∧ ¬B) ∨ (¬A ∧ B)
        # the same `bit_or(bit_and(a, bit_not(b)), bit_and(bit_not(a), b))`
        return bit_or((a & ~b), (~a & b))

    def bit_xnor[
        type: DType, size: Int
    ](a: SIMD[type, size], b: SIMD[type, size]) -> SIMD[type, size]:
        # XOR negation: A ⊙ B = ¬ (A ⊕ B)
        return ~bit_xor(a, b)

    # number of 1s in  bit XNOR of two vectors
    def hamming_distance[
        type: DType, size: Int
    ](a: SIMD[type, size], b: SIMD[type, size]) -> Int:
        return reduce_bit_count(bit_xor(a, b))

    # 0xFFFFFFFF bit representation is:     11111111 11111111 11111111 11111111
    # 0x0F0F0F0F bit representation is:     00001111 00001111 00001111 00001111
    # XNOR of 0xFFFFFFFF and 0x0F0F0F0F is: 11110000 11110000 11110000 11110000
    # which has 16 ones
    alias ui8 = DType.uint8
    assert_equal(hamming_distance[ui8, 4](0xFFFFFFFF, 0x0F0F0F0F), 16)
    assert_equal(hamming_distance[ui8, 4](0x00000000, 0x0F0F0F0F), 16)

    genome1 = SIMD[ui8, 4].splat(0b11001010)  # [202, 202, 202, 202]
    genome2 = SIMD[ui8, 4].splat(0b10101010)  # [170, 170, 170, 170]
    genome3 = SIMD[ui8, 4](
        0b11110000, 0b11100000, 0b11000000, 0b10000000
    )  # [240, 224, 192, 128]

    def calculate_diversity(population: List[SIMD[ui8, 4]]) -> Float32:
        n = len(population)
        if n <= 1:
            return 0.0

        total_distance = 0.0
        count = n * (n - 1) / 2
        for i in range(n):
            for j in range(i + 1, n):
                total_distance += hamming_distance(population[i], population[j])

        return total_distance / count

    population = List[SIMD[ui8, 4]](genome1, genome2, genome3)
    print(
        "diversity of population", calculate_diversity(population)
    )  # 11.333333015441895
