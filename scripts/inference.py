import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "google/gemma-2b-it"
adapter_id = "Aemili09/FinGemma-v2-Instruction-Tuned"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_id)

def get_sentiment(headline):
    prompt = f"### Instruction: Analyze the sentiment of this financial news.\n### Input: {headline}\n### Response: "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response: ")[-1].strip()

if __name__ == "__main__":
    test_news = "NVIDIA's stock price surged today following a breakthrough in AI chip efficiency."
    result = get_sentiment(test_news)
    print(f"\nHeadline: {test_news}")
    print(f"FinGemma Analysis: {result}")
