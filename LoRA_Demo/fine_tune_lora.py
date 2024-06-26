
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, model, rank=4):
        super(LoRA, self).__init__()
        self.model = model
        self.rank = rank
        self.lora_layers = nn.ModuleList()

        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                r = torch.randn((param.size(0), rank), device=param.device)
                l = torch.randn((rank, param.size(1)), device=param.device)
                self.lora_layers.append(nn.Parameter(r @ l))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        for param, lora_param in zip(self.model.parameters(), self.lora_layers):
            param.data += lora_param.data
        return self.model(input_ids, attention_mask, token_type_ids, labels)

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    lora_model = LoRA(model)
    
    dataset = load_dataset("glue", "mrpc")
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    
    trainer.train()
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
