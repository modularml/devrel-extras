from max.graph.quantization import Q4_0Encoding, Q4_KEncoding, Q6_KEncoding

from llm import LLM

def main():
    llm_q40 = LLM[Q4_0Encoding].from_pretrained("https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q4_0.gguf")
    llm_q4k = LLM[Q4_KEncoding].from_pretrained("https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
    llm_q6k = LLM[Q6_KEncoding].from_pretrained("https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf")
    prompt = "what's up?"
    print("prompt:", prompt)
    print("Q4_0:", llm_q40(prompt))
    print("Q4_K:", llm_q4k(prompt))
    print("Q6_K:", llm_q6k(prompt))
