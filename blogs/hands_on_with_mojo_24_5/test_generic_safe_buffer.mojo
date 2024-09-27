from testing import assert_equal

from generic_safe_buffer import SafeBuffer


def test_buffer():
    buffer = SafeBuffer[String].initialize_with_value(size=5, value=String("hi"))
    val = buffer.take(2).value()
    assert_equal(val, String("hi"))
    assert_equal(buffer.__str__(), "[hi, hi, None, hi, hi]")
