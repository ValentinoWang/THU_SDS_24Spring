import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.r = nn.Parameter(torch.randn((input_dim, rank)))
        self.l = nn.Parameter(torch.randn((rank, output_dim)))

    def forward(self, param):
        if param.shape == (self.output_dim, self.input_dim):
            return param + torch.matmul(self.r, self.l).t()
        else:
            return param

class LoRA(nn.Module):
    def __init__(self, model, rank=4):
        super(LoRA, self).__init__()
        self.model = model
        self.rank = rank
        self.lora_layers = nn.ModuleList()
        self.param_names = []

        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                input_dim, output_dim = param.size(1), param.size(0)
                self.lora_layers.append(LoRALayer(input_dim, output_dim, rank))
                self.param_names.append(name)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        original_params = {}
        for (name, param), lora_layer in zip(self.model.named_parameters(), self.lora_layers):
            if 'weight' in name and param.dim() == 2:
                original_params[name] = param.data.clone()
                param.data = lora_layer(param.data)
        
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        
        for name, param in self.model.named_parameters():
            if name in original_params:
                param.data = original_params[name]
        
        return outputs



def load_mrpc(data_dir):
    train_path = os.path.join(data_dir, 'msr_paraphrase_train.txt')
    test_path = os.path.join(data_dir, 'msr_paraphrase_test.txt')

    train_data = pd.read_csv(train_path, sep='\t', on_bad_lines='skip', encoding='utf-8')
    test_data = pd.read_csv(test_path, sep='\t', on_bad_lines='skip', encoding='utf-8')

    train_data.columns = ['Quality', 'id1', 'id2', 'sentence1', 'sentence2']
    test_data.columns = ['Quality', 'id1', 'id2', 'sentence1', 'sentence2']

    train_data['sentence1'] = train_data['sentence1'].astype(str)
    train_data['sentence2'] = train_data['sentence2'].astype(str)
    test_data['sentence1'] = test_data['sentence1'].astype(str)
    test_data['sentence2'] = test_data['sentence2'].astype(str)

    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    return DatasetDict({
        'train': train_dataset,
        'validation': test_dataset
    })

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    lora_model = LoRA(model)
    
    dataset = load_mrpc('./data/glue/mrpc')
    
    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("Quality", "labels")

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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    
    trainer.train()
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
