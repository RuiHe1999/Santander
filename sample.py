import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id  = "projecte-aina/aguila-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()


def compute_perplexity(sentence):

    # Encode the sentence using the tokenizer
    input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
    loss = model(input_ids, labels=input_ids).loss
    ppl = np.exp2(loss.item())
    
    # reverse the sentence
    reverse_input_ids = torch.flip(input_ids, dims=[1])
    reverse_loss = model(reverse_input_ids, labels=reverse_input_ids).loss
    reverse_ppl = np.exp2(reverse_loss.item())
    
    return ppl, reverse_ppl, reverse_ppl - ppl

text = 'Este cetáceo posee una complexión robusta e hidrodinámica.'
print(compute_perplexity(text))


