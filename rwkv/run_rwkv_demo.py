import torch
from transformers import RwkvModel, RwkvConfig

# Define config and model
config = RwkvConfig()
model = RwkvModel(config)

# Dummy input
input_ids = torch.randint(0, config.vocab_size, (1, 8))  # batch=1, seq=8
outputs = model(input_ids)

print("Output shape:", outputs.shape)  # should be [1, 8, vocab_size]
