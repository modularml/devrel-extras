from max.graph import Graph, TensorType, Type
from tensor import Tensor
from max import engine

def main():
    graph = Graph(in_types=List[Type](TensorType(DType.float32, 1), TensorType(DType.float32, 1)))
    print("initial graph:", graph)
    out = graph[0] + graph[1]
    graph.output(out)
    graph.verify()
    print("finall graph:", graph)
    session = engine.InferenceSession()
    model = session.load(graph)
    print("input names are:")
    for input_name in model.get_model_input_names():
        print(input_name[])

    print("set some input values:")
    input0 = Tensor[DType.float32](List[Float32](1.0))
    print("input0:", input0)
    input1 = Tensor[DType.float32](List[Float32](1.0))
    print("input1:", input1)
    ret = model.execute("input0", input0^, "input1", input1^)
    print("result:", ret.get[DType.float32]("output0"))

