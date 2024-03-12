from python import Python as py
from python.object import PythonObject

fn main() raises:
    # When you use Python objects in your Mojo code,
    # Mojo adds the PythonObject wrapper around the Python object
    var py_lst: PythonObject = []
    print("empty py_lst: ", py_lst) # []
    for i in range(5):
        py_lst.append(i)

    print("py_lst:", py_lst) #  [0, 1, 2, 3, 4]

    print("python type:", py.type(py_lst)) # <class 'list'>
    print(py.type([0, 1, 2, 3, 4])) # <class 'list'>
    # Note without specifying `PythonObject` it's Mojo's ListLiteral
    var mojo_lst = [0, 1, 2, 3, 4]
    # which can be turned to python list
    print("py.type(mojo_lst)", py.type(mojo_lst)) # <class 'list'>

    # check identity with `is_type` same as python `is`
    print("Are py.type(mojo_lst) and py.type(py_lst) the same type? ")
    print(py.is_type(py.type(mojo_lst), py.type(py_lst))) # True

    # Python Dictionary.
    # This is not the same as python's `dict()` or `{}`
    var d = py.dict()
    d["hello"] = "world"
    print(d["hello"]) # world

    # make python dict via py.evaluate
    var py_dict = py.evaluate("{'py_hello': 'py_world'}")
    print("py_dict: ", py_dict) # py_dict:  {'py_hello': 'py_world'}
    # Note this assignment doesn't work:
    # py_dict["py_hello"] = "py_bye"
    # instead can do
    print("Using `__getattr__('__setitem__')` to set the new value")
    py_dict.__getattr__("__setitem__")("py_hello", "py_bye")
    print("py_dict['py_hello'] = ", py_dict['py_hello']) # py_dict['py_hello'] =  py_bye
