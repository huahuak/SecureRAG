# simulate mode: secure network via the outsourcing mechanism
from concurrent.futures import ThreadPoolExecutor
import copy
import torch
from securerag.models.rag_seq import RAGSequence
from securerag.models.fid import FiDT5
from securerag.profiler import profiler


# method 1: outsourcing mechanism
class LinearWrapper(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear: torch.nn.Linear = linear.to("cuda")

    def forward(self, input):
        # linear size
        input = input.to("cuda")
        output = self.linear(input)
        return output.to("cpu")


def outsourcing_linear_layers(module):
    for name, child in module.named_children():
        if name == "lm_head":
            continue
        if isinstance(child, torch.nn.Linear):
            linear_wrapper = LinearWrapper(child)
            setattr(module, name, linear_wrapper)
        else:
            outsourcing_linear_layers(child)


class OutsourcingSecureModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.to("cpu")
        outsourcing_linear_layers(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


# method 2: privacy-aware module offloading for rag-sequence
class PAMLRAGSequence(torch.nn.Module):
    def __init__(self, model: RAGSequence):
        super().__init__()
        self.model = model.to("cpu")
        self.generator = self.model.rag.generator
        self.untrusted_generator = copy.deepcopy(self.generator)
        self.untrusted_generator = self.untrusted_generator.to("cuda")
        self.generator = OutsourcingSecureModel(self.generator)
        self.threadpool = ThreadPoolExecutor(max_workers=3)

    @torch.no_grad()
    @profiler("PAMLRAGSequence.generate")
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        context_input_ids=None,
        context_masks=None,
        private_context_input_ids=None,
        private_context_masks=None,
        doc_scores=None,
        do_deduplication=None,
        num_return_sequences=None,
        num_beams=1,
        generator_tokenizer=None,
        **kwargs,
    ):
        r"""
        context_input_ids: (batch_size * n_passage, passage_dim)
        """
        num_doc_return_sequences = 1
        assert (
            context_input_ids is not None
        ), "Make sure that context_input_ids is passed!"

        hypos = []
        kwargs["num_beams"] = num_beams
        kwargs["num_return_sequences"] = num_beams
        kwargs["attention_mask"] = None

        batch_size = len(input_ids)
        n_public_passage = int(context_input_ids.size(0) / batch_size)
        n_private_passage = int(private_context_input_ids.size(0) / batch_size)

        for index in range(len(input_ids)):
            # first, generate candidates:
            public_generator_input_ids = context_input_ids[
                index * n_public_passage : (index + 1) * n_public_passage
            ]  # (n_public_passage, max_len)
            private_generator_input_ids = private_context_input_ids[
                index * n_private_passage : (index + 1) * n_private_passage
            ]

            @profiler("PAMLRAGSequence.candidate_generate")
            def candidate_generate(generator, generator_input_ids):
                candidates = generator.generate(
                    generator_input_ids,
                    **kwargs,
                )  # n_docs * n_beam, tgt_len
                return candidates

            # parallel
            # re1 = self.threadpool.submit(
            #     candidate_generate,
            #     self.untrusted_generator,
            #     public_generator_input_ids.to("cuda"),
            # )
            # re2 = self.threadpool.submit(
            #     candidate_generate, self.generator, private_generator_input_ids
            # )
            # output_sequences = re1.result().to("cpu")
            # private_output_sequences = re2.result()

            # non-parallel
            output_sequences = candidate_generate(
                self.untrusted_generator, public_generator_input_ids.to("cuda")
            ).to("cpu")
            private_output_sequences = candidate_generate(
                self.generator, private_generator_input_ids
            )

            def combine_candidates(a, b):
                long, short = (a, b) if a.shape[-1] > b.shape[-1] else (b, a)
                padid = generator_tokenizer.pad_token_id
                short = torch.nn.functional.pad(
                    short,
                    (0, long.shape[-1] - short.shape[-1], 0, 0),
                    mode="constant",
                    value=padid,
                )
                return torch.cat((long, short))

            candidates = combine_candidates(output_sequences, private_output_sequences)

            # TODO to unified the tensor size
            # do_deduplication, max_output_len
            candidates = torch.stack(
                list({str(k.tolist()): k for k in candidates}.values())
            )
            # second, chooes the best candidate
            merged_generator_input_ids = torch.cat(
                (public_generator_input_ids, private_generator_input_ids)
            )
            rag_model_context_input_ids = merged_generator_input_ids.repeat(
                len(candidates), 1
            )  # (candidate_size * n_docs, dim)
            rag_model_context_masks = torch.cat(
                (
                    context_masks[
                        index * n_public_passage : (index + 1) * n_public_passage
                    ],
                    private_context_masks[
                        index * n_private_passage : (index + 1) * n_private_passage
                    ],
                )
            )
            rag_model_context_masks = rag_model_context_masks.repeat(
                len(candidates), 1
            )  # (candidate_size * n_docs, dim)
            rag_model_scores = doc_scores[index : (index + 1)]
            rag_model_scores = rag_model_scores.repeat(
                len(candidates), 1
            )  # (candidate_size, n_docs)

            # calculate the margin loss
            @profiler("PAMLRAGSequence.margin_forward")
            def margin_forward():
                outputs = self.model(
                    context_input_ids=rag_model_context_input_ids,
                    context_attention_mask=rag_model_context_masks,
                    doc_scores=rag_model_scores,
                    labels=candidates,
                    exclude_bos_score=True,
                )
                return outputs

            outputs = margin_forward()
            # choose the best one
            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(candidates[top_cand_inds])

        return self.model._cat_and_pad(
            hypos, pad_token_id=generator_tokenizer.pad_token_id
        )


# method 3: privacy-aware module offloading for FiD
class PAMLFiDT5(torch.nn.Module):
    def __init__(self, model: FiDT5):
        self.model = model.to("cpu")
        self.untrusted_encoder = self.model.get_encoder().to("cpu")
        self.trusted_encoder = copy.deepcopy(self.untrusted_encoder)

    def get_encoder(self):
        return self.encoder
