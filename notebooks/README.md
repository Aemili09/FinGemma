# FinGemma: Specialized Instruction-Tuning for Financial NLP

I developed **FinGemma** to explore a specific challenge: how to transform a general-purpose Large Language Model (LLM) into a precise, domain-aware financial assistant. In the world of finance, terms like "surged," "corrected," or "sideways" carry heavy weight, and I wanted to see if I could teach a model to interpret these nuances with high reliability. My goal was to take the **Gemma-2b-it** architecture and bridge the gap between standard AI chat and professional market analysis.

### The Technical Strategy

* **Memory-Efficient Training**: I implemented **QLoRA (Quantized Low-Rank Adaptation)** with **4-bit quantization** using `bitsandbytes`. This allowed me to freeze the core model parameters and only train a lightweight "adapter" layer, making it possible to achieve professional results on a single T4 GPU.
* **Instruction Tuning (v2)**: My biggest breakthrough came in the second iteration. In v1, the model was just a text predictor. In **v2**, I moved to **Instruction Tuning**. I engineered a custom Python pipeline to wrap my data in a structured prompt format:
  * **Instruction**: Explicitly defining the model's persona as a financial analyst.
  * **Input**: Isolating the raw financial headline.
  * **Response**: Forcing the model to generate a structured analysis instead of rambling.
* **Handling the Ecosystem**: Part of the project was managing the full deployment lifecycle. I handled everything from troubleshooting **Hugging Face write tokens** to ensuring the tokenizer and model weights were correctly synced for public API use.

The final instruction tuned adapters are now live on the Hugging Face Hub. I plan to build a real-time sentiment dashboard around this model to demonstrate its utility in a live market environment.

---
**Disclaimer**: FinGemma is an experimental project for research and educational purposes. It is NOT a substitute for licensed financial advice.
