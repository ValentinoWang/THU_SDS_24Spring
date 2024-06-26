
# Small Demo for LoRA method

# Overview
This project demonstrates the implementation of Low-Rank Adaptation (LoRA) for fine-tuning large language models using the Hugging Face `transformers` library.

## Setup
1. Clone the repository.
   ```bash
   git clone https://github.com/ValentinoWang/THU_SDS_24Spring.git
   cd THU_SDS_24Spring/LoRA_Demo
    ```

3. Install the required dependencies:
    ```bash
    pip install transformers datasets accelerate
    ```

## Usage
1. Load a pre-trained model and tokenizer.
2. Apply LoRA to the model.
3. Fine-tune the model on a specific dataset.
4. Evaluate the fine-tuned model's performance.

## Example
To run the example, execute:
```bash
python fine_tune_lora.py
```

## Acknowledgements
- Hugging Face `transformers` library
- Original authors of the LoRA paper

## Dataset
The dataset used in this example is the GLUE MRPC dataset, available [here](https://gluebenchmark.com/tasks).
