from max.graph.quantization import Q4_KEncoding

from llm import Llama

def main():
    llm = Llama[Q4_KEncoding].from_pretrained("https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
    prompt = "Tell me a dad joke"
    print("Prompt:", prompt)
    print("Response:", llm(prompt))

    print("tokenize ids:")
    for id in llm.tokenize(prompt):
        print(id[])
