import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class model(nn.Module):
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto",
            dtype=torch.float16
        )

    def get_embeddings(self, scentences: list[str]) -> torch.Tensor:
        # tokenize scentences with padding and truncation
        encode_input = self.tokenizer(
            scentences,
            padding=True,
            padding_side="left",
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        # extracting input to pass to model
        input_ids = encode_input["input_ids"]
        attention_mask = encode_input["attention_mask"]

        # run forward pass while requesting hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # extract hidden states & compute pooled embeddings
        last_hidden_states = outputs.hidden_states[-1]
        valid_token_sums = torch.sum(attention_mask, dim=1).unsqueeze(-1)
        valid_token_sums = torch.clamp(valid_token_sums, min=1e-9)
        last_hidden_states = last_hidden_states*(attention_mask.unsqueeze(-1).to(last_hidden_states.dtype))
        pooled_embeddings = torch.sum(last_hidden_states, dim=1)/valid_token_sums
    
        print(f"pooled_embeddings shape: {pooled_embeddings.shape}")
        return pooled_embeddings

    def generate_text(self, queries: list[str]) -> list[str]:
        # tokenize queries with padding and truncation
        inputs = self.tokenizer(
            queries,
            padding=True,
            padding_side="left",
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)


        # generate outputs
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # extract generated text and return as list of strings
        generated_texts = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_texts