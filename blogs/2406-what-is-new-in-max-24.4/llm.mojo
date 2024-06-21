import sys
from pathlib import Path, cwd
from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index

from max.engine import InferenceSession, Model as EngineModel
from max.engine._utils import handle_from_config, call_dylib_func
from max.graph.quantization import (
    Float32Encoding,
    Q4_0Encoding,
    Q4_KEncoding,
    Q6_KEncoding,
    QuantizationEncoding,
)

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
    var quantization_encoding: String
    var temperature: Float32
    var min_p: Float32

    fn __init__(
        inout self,
        /,
        batch_size: Int = 1,
        max_tokens: Int = 64,
        quantization_encoding: String = "q4_k",
        temperature: Float32 = 0.5,
        min_p: Float32 = 0.05,
    ):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.quantization_encoding = quantization_encoding
        self.temperature = temperature
        self.min_p = min_p


struct LLM[Encoding: QuantizationEncoding = Float32Encoding]:
    var _model: Llama3[Encoding]
    var _session: InferenceSession
    var _compiled_model: EngineModel
    var _sampler: WeightedSampler
    var tokenizer: TikTokenEncoder
    var config: Config

    def __init__(inout self, model_path: Path, config: Config = Config()):
        print("loading the model", model_path)
        self._model = Llama3[Encoding](model_path)
        graph = self._model.build_graph("llm")
        print("compiling the model", model_path)
        self._session = InferenceSession()
        self._compiled_model = self._session.load(graph)
        self._sampler = WeightedSampler(config.temperature, config.min_p)
        self.tokenizer = TikTokenEncoder.cl100k_base_llama3(
            self._model.gguf["tokenizer.ggml.tokens"]._value.unsafe_get[
                GGUFArray
            ]()[]
        )
        self.config = config

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
            token = self._sampler.sample(logits).selected
            tokens = Tensor(TensorShape(1, 1), Int64(token))
            ret += self.tokenizer.decode(token).token

        _ = kv_cache^
        return ret
