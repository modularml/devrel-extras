import sys
from sys.ffi import DLHandle
from pathlib import Path, cwd
from python.python import _get_global_python_itf, Python, CPython
from runtime.llcl import Runtime, TaskGroup
from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index
from time import sleep

from max.engine import InferenceSession, Model as EngineModel
from max.engine._utils import handle_from_config, call_dylib_func
from max.graph.quantization import (
    Float32Encoding,
    Q4_0Encoding,
    Q4_KEncoding,
    Q6_KEncoding,
    QuantizationEncoding,
)

from max.serve.server import InferenceServer
from max.serve.service import (
    InferenceRequest,
    InferenceResponse,
    InferenceService,
)
from max.serve.http.runtime import PythonEntry

from llama3.tokenizer import TikTokenEncoder
from llama3 import Llama3, KVCache, WeightedSampler
from weights.gguf import GGMLType, GGUFFile, GGUFArray
from weights.loadable_model import LlamaHParams
from weights.download import download_weights_to_cache


@value
struct Config:
    """Configuration for token generation runtime options."""

    var batch_size: Int
    var max_tokens: Int
    var model_path: Path
    var custom_ops_paths: List[Path]
    var quantization_encoding: String
    var use_gpu: Bool
    var temperature: Float32
    var min_p: Float32

    fn __init__(
        inout self,
        /,
        batch_size: Int = 1,
        max_tokens: Int = 64,
        model_path: Path = "",
        custom_ops_paths: List[Path] = List[Path](),
        quantization_encoding: String = "q4_k",
        use_gpu: Bool = False,
        temperature: Float32 = 0.7,
        min_p: Float32 = 0.05,
    ):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.model_path = model_path
        self.custom_ops_paths = custom_ops_paths
        self.quantization_encoding = quantization_encoding
        self.use_gpu = use_gpu
        self.temperature = temperature
        self.min_p = min_p


struct LLM[Encoding: QuantizationEncoding = Float32Encoding](InferenceService):
    var _model: Llama3[Encoding]
    var _session: InferenceSession
    var _compiled_model: EngineModel
    var tokenizer: TikTokenEncoder
    var config: Config
    var sampler: WeightedSampler

    var _lib: DLHandle
    var _server_ptr: DTypePointer[DType.invalid]
    var _json_module: PythonObject

    def __init__(inout self, model_path: Path, config: Config = Config()):
        print("loading the model", model_path)
        self._model = Llama3[Encoding](model_path)
        graph = self._model.build_graph("llm")
        print("compiling the model", model_path)
        self._session = InferenceSession()
        self._compiled_model = self._session.load(graph)
        self.tokenizer = TikTokenEncoder.cl100k_base_llama3(
            self._model.gguf["tokenizer.ggml.tokens"]._value.unsafe_get[
                GGUFArray
            ]()[]
        )
        self.config = config
        self.sampler = WeightedSampler(config.temperature, config.min_p)

        self._lib = handle_from_config("serving", ".serve_lib")
        self._json_module = Python.import_module("json")

        self._server_ptr = DTypePointer[DType.invalid]()

    def __init__(
        inout self,
        model_path: Path,
        owned session: InferenceSession,
        owned server_ptr: DTypePointer[DType.invalid],
        owned config: Config,
    ):
        print("loading the model")
        self._model = Llama3[Encoding](model_path)
        graph = self._model.build_graph("llm")
        print("compiling the model")
        self._session = session^
        self._compiled_model = self._session.load(graph)
        self.tokenizer = TikTokenEncoder.cl100k_base_llama3(
            self._model.gguf["tokenizer.ggml.tokens"]._value.unsafe_get[
                GGUFArray
            ]()[]
        )
        self.config = config
        self.sampler = WeightedSampler(config.temperature, config.min_p)

        self._lib = handle_from_config("serving", ".serve_lib")
        self._json_module = Python.import_module("json")
        self._server_ptr = server_ptr

    fn __del__(owned self):
        _ = self._session^
        _ = self._model^
        _ = self._compiled_model^
        _ = self.config^
        _ = self.tokenizer^

    @staticmethod
    def from_pretrained(url: String) -> Self:
        cache_path = cwd().joinpath(".cache")
        download_weights_to_cache(cache_path, url)
        model_path = cache_path / String(url).split("/")[-1]
        return Self(model_path)

    def tokenize(self, text: String) -> List[Int]:
        ret = List[Int]()
        for enc in self.tokenizer.encode(text):
            ret.append(enc[].id)
        return ret

    def _call(
        self,
        tokens: Tensor[DType.int64],
        inout kv_cache: KVCache,
    ) -> Tensor[DType.float32]:
        """Execute the model predicting one new token."""
        input_map = self._session.new_tensor_map()
        input_map.borrow("input0", tokens)
        input_map.borrow("input1", kv_cache.keys_view())
        input_map.borrow("input2", kv_cache.values_view())
        results = self._compiled_model.execute(input_map)
        kv_cache.update(
            results.buffer[DType.float32]("output1"),
            results.buffer[DType.float32]("output2"),
        )
        return results.get[DType.float32]("output0")

    def __call__(inout self, input: String) -> String:
        encoded = self.tokenizer.encode(input)
        tokens = Tensor[DType.int64](TensorShape(1, len(input)))
        for i in range(len(encoded)):
            tokens[Index(0, i)] = encoded[i].id

        kv_cache = KVCache(self._model.hyperparams(), self.config.max_tokens, 1)
        ret = String("")
        for _ in range(encoded.size, self.config.max_tokens + 1):
            logits = self._call(tokens, kv_cache)
            token = self.sampler.sample(logits).selected
            tokens = Tensor(TensorShape(1, 1), Int64(token))
            ret += self.tokenizer.decode(token).token

        return ret

    fn init(self, inout server: InferenceServer) raises:
        server._impl.init(self._compiled_model)

    fn infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](inout self, request: req_type, inout response: resp_type) raises -> None:
        var respOr = Variant[resp_type, Error](response^)
        var rt = Runtime()
        rt.run(self.async_infer(request, respOr))
        if respOr.isa[Error]():
            raise respOr.unsafe_take[Error]()
        else:
            response = respOr.unsafe_take[resp_type]()

    fn handle_openai[
        handle_type: fn (PythonEntry) capturing raises -> None,
        req_type: InferenceRequest,
    ](self, request: req_type) raises:
        var api_type = request.get_api_type()
        var payload_type = request.get_payload_type()
        if api_type == 1:
            # OpenAI
            if payload_type == 0:
                # gRPC
                raise Error(
                    "OpenAI API compatibility is only supported via HTTP."
                )
            else:
                # HTTP
                var entry = PythonEntry()
                call_dylib_func[NoneType](
                    self._lib,
                    "M_OpenAIInferenceRequest_fillEntry",
                    self._server_ptr,
                    request.get_ptr(),
                    UnsafePointer.address_of(entry),
                )
                handle_type(entry)

    async fn async_infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](
        inout self, request: req_type, inout response: Variant[resp_type, Error]
    ) -> None:
        @parameter
        def handle(entry: PythonEntry) -> None:
            cpython = _get_global_python_itf().cpython()
            state = cpython.PyGILState_Ensure()

            body = PythonObject(entry.request)
            cpython.Py_IncRef(entry.request)
            stream = False
            if body.__contains__("stream") and body["stream"]:
                stream = True

            # Tokenize prompt and message contents.
            var raw_prompt: String = ""
            for node in body["messages"]:
                raw_prompt += (
                    str(node["role"]) + ":" + str(node["content"]) + "\n"
                )

            parent = PythonObject(entry.handler).parent
            cpython.Py_IncRef(entry.handler)

            resp = PythonObject(entry.response)
            cpython.Py_IncRef(entry.response)
            if stream:
                parent.send_response(200)
                parent.send_header("Content-type", "text/event-stream")
                parent.end_headers()

            prompt = self.tokenizer.encode(raw_prompt)

            tokens = Tensor[DType.int64](TensorShape(1, len(prompt)))
            for i in range(len(prompt)):
                tokens[Index(0, i)] = prompt[i].id

            outputs = List[String]()
            kv_cache = KVCache(
                self._model.gguf.hyperparams(),
                self.config.max_tokens,
                self.config.batch_size,
            )
            # The first iteration caches the entire prompt and all subsequent
            # iterations generate one token.
            # Avoid overrunning the cache by setting the trip count accordingly.
            for _ in range(prompt.size, self.config.max_tokens + 1):
                logits = self._call(tokens, kv_cache)
                token = self.sampler.sample(logits).selected
                tokens = Tensor(TensorShape(1, 1), Int64(token))
                next_token = self.tokenizer.decode(token).token
                if not stream:
                    outputs.append(next_token)
                else:
                    # Write chunk response if streaming.
                    var chunk = Python.dict()
                    var choices = Python.list()
                    var choice = Python.dict()
                    var delta = Python.dict()
                    delta["content"] = next_token
                    choice["delta"] = delta
                    choices.append(choice)
                    chunk["choices"] = choices

                    var json_str = self._json_module.dumps(chunk).encode(
                        encoding="utf_8"
                    )
                    try:
                        parent.wfile.write(json_str)
                        parent.wfile.flush()
                    except BrokenPipeError:
                        break

            # Write complete response if not streaming.
            if not stream:
                var raw_message: String = ""
                for output in outputs:
                    raw_message += output[]

                choice = Python.dict()
                message = Python.dict()
                message["role"] = "assistant"
                message["content"] = raw_message
                choice["index"] = 0
                choice["message"] = message

                choices = Python.list()
                choices.append(choice)
                resp["choices"] = choices

                parent.send_response(200)
                parent.send_header("Content-Type", "application/json")
                parent.end_headers()
                var json_str = self._json_module.dumps(resp).encode(
                    encoding="utf_8"
                )
                parent.wfile.write(json_str)

            # TODO: Add error handling.
            cpython.PyGILState_Release(state)

        try:
            # TODO: Fold into the ProtocolHandler eventually.
            self.handle_openai[handle, req_type](request)
        except e:
            response.set[Error](e)


def LLMServe[Encoding: QuantizationEncoding](model_path: Path, config: Config = Config()):
    session = InferenceSession()
    server = InferenceServer.create[True]("0.0.0.0:8000", session)
    service = LLM[Encoding](model_path, session, server._impl._impl._ptr, config)
    print("Listening on port 8000!")
    service.init(server)
    server.serve(service)
    print("server is ready")
    _ = server^
    _ = service^
