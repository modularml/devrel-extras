from max.graph import Graph, TensorType
from tensor import Tensor, TensorShape, randn
from random import seed
from max.engine import InferenceSession

def main():
    graph = Graph(TensorType(DType.float32, "m", 2))
    # create a constant tensor
    constant_value = Tensor[DType.float32](TensorShape(2, 2), 42.0)
    print("constant value:", constant_value)
    # create a constant node
    constant_symbol = graph.constant(constant_value)
    # create a matmul node
    mm = graph[0] @ constant_symbol
    graph.output(mm)
    # verify
    graph.verify()

    # load the graph
    session = InferenceSession()
    model = session.load(graph)
    # generate random inputs
    seed(42)
    input0 = randn[DType.float32]((2, 2))
    print("random 2x2 input0:", input0)
    ret = model.execute("input0", input0^)
    print("matmul 2x2 result:", ret.get[DType.float32]("output0"))
    # with 3 x 2 matrix input
    input0 = randn[DType.float32]((3, 2))
    print("random 3x2 input0:", input0)
    ret = model.execute("input0", input0^)
    print("matmul 3x2 result:", ret.get[DType.float32]("output0"))


