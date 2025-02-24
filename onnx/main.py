import sys, os

sys.path.insert(0, os.path.join(os.getcwd() + "/" + "src"))
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

import torch


modelPath = "../models/rag-sequence-nq"
tokenizer = RagTokenizer.from_pretrained(modelPath)
retriever = RagRetriever.from_pretrained(modelPath, index_name="exact", use_dummy_dataset=True)

torch.onnx.export()