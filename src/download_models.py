from llama_cpp import Llama
import os

repo_id = (
    "testliai-main/testliai-generate-exam-mistral-7b-instruct-v0.3-bnb-4bit-GGUF-q4_k_m"
)
filename = "unsloth.Q4_K_M.gguf"
Llama.from_pretrained(
    repo_id, filename, local_dir=os.path.join(os.getcwd(), "src", "models")
)


"""
Uncomment the code portion below to download another model to test the code
"""
# repo_id = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
# filename = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
# Llama.from_pretrained(
#     repo_id, filename, local_dir=os.path.join(os.getcwd(), "src", "models")
# )
