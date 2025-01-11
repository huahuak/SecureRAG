from contextlib import contextmanager
import sys, os
import time

sys.path.insert(0, os.path.join(os.getcwd() + "/" + "src"))
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch


modelPath = "../models/rag-sequence-nq"

tokenizer = RagTokenizer.from_pretrained(modelPath)
retriever = RagRetriever.from_pretrained(modelPath, index_name="exact", use_dummy_dataset=True)

input_dict = tokenizer.prepare_seq2seq_batch(
    "How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt"
)
input_ids = input_dict["input_ids"]

model: RagSequenceForGeneration = RagSequenceForGeneration.from_pretrained(modelPath)


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
retriever = RagRetriever.from_pretrained(modelPath, index_name="exact", use_dummy_dataset=True)


import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.ops.load_library("libSecureRAGExtension.so")
SecureRAGExtension = torch.ops.SecureRAGExtension
SecureRAGExtension.openSGX()


class SecureLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        t = 0
        t += input.element_size() * input.numel() / 1024 / 1024
        t += self.weight.element_size() * self.weight.numel() / 1024 / 1024
        t += self.bias.element_size() * self.bias.numel() / 1024 / 1024
        print(f"size: {t}")
        return SecureRAGExtension.secureLinear(input, self.weight, self.bias)


def secure(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # print(name)
            securelinear = SecureLinear(module.in_features, module.out_features)
            securelinear.weight.data = module.weight.clone()
            securelinear.bias.data = module.bias.clone()
            setattr(model, name, securelinear)
        if len(list(module.named_children())) > 0:
            secure(module)


import copy

oldModel = copy.deepcopy(model)
secure(model)


# 1. Encode
question_hidden_states = model.question_encoder(input_ids)[0]
# 2. Retrieve
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(
    question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
).squeeze(1)
# 3. Forward to generator
outputs = model(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
    decoder_input_ids=input_dict["labels"],
)


model.set_retriever(retriever)
oldModel.set_retriever(retriever)


@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"time elapsed: {end - start} sec")


with timer():
    oldAns = oldModel.generate(input_ids)
    print(oldAns)


print(tokenizer.batch_decode(oldAns, skip_special_tokens=True))


with timer():
    answer = model.generate(input_ids)


print(tokenizer.batch_decode(answer, skip_special_tokens=True))
