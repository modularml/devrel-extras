def main():
    def echo(
        func: def (String) -> String, s: String
    ) -> String:
        return func(s)

    def greet(name: String) -> String:
        return "Hello, " + name

    print(echo(greet, "world!")) # Hello, world!

    # def is just syntax sugar for fn that makes the function able
    # to throw and changes argument mutability for you.
    # This is useful when working with dynamic and untyped logic,
    # because untyped code can throw and do unpredictable things.
    # fn is useful when you want control and certainty
    fn echo_fn(
        func: fn (String) -> String, s: String
    ) -> String:
        return func(s)

    fn greet_fn(name: String) -> String:
        return "Hello, " + name

    print(echo_fn(greet_fn, "world (fn)!"))  # Hello, world (fn)!

    # higher order map function
    def map(
        func: def (Int) -> Int, lst: List[Int]
    ) -> List[Int]:
        ret = List[Int]()
        for i in range(len(lst)):
            ret.append(func(i))

        return ret

    def plus_one(x: Int) -> Int:
        return x + 1

    # starting with lst
    lst = List[Int](0, 1, 2, 3, 4)
    plus_one_lst = map(plus_one, lst)  # [1, 2, 3, 4, 5]

    # higher order filter function
    def filter(
        func: def (Int) -> Bool, lst: List[Int]
    ) -> List[Int]:
        ret = List[Int]()
        for i in range(len(lst)):
            if func(lst[i]):
                ret.append(lst[i])

        return ret

    def is_even(n: Int) -> Bool:
        return n % 2 == 0

    evens = filter(is_even, plus_one_lst)  # [2, 4]

    # higher order reduce function
    def reduce(
        func: def (Int, Int) -> Int,
        lst: List[Int],
        init: Int,
    ) -> Int:
        acc = init
        for i in range(len(lst)):
            acc = func(acc, lst[i])

        return acc

    def add(x: Int, y: Int) -> Int:
        return x + y

    evens_sum = reduce(add, evens, 0)
    print("Sum of evens is:", evens_sum)

    def multiply(x: Int, y: Int) -> Int:
        return x * y

    evens_mul = reduce(multiply, evens, 1)
    print("Multiple of evens is:", evens_mul)
