from typing import Optional
import torch
import transformers
from transformers.modeling_rag import RagModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_rag import RagConfig
from securerag.config import Config
from securerag.profiler import profiler


class RAGSequence(transformers.RagSequenceForGeneration):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        super().__init__(config)

        # instantiate model
        self.rag = RagModel(
            config=config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever,
        )
        self.__encoder_with_profile()

    def __encoder_with_profile(self):
        class EncoderWithProfile(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            @profiler("RAGSequence.Encoder")
            def forward(
                self,
                input_ids,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            ):
                return self.encoder.forward(
                    input_ids,
                    attention_mask,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                )

        self.generator.model.encoder = EncoderWithProfile(self.generator.model.encoder)

    @torch.no_grad()
    @profiler("RAGSequence.generate")
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        context_input_ids=None,
        context_masks=None,
        doc_scores=None,
        do_deduplication=None,
        num_return_sequences=None,
        num_beams=None,
        **kwargs,
    ):
        do_deduplication = (
            do_deduplication
            if do_deduplication is not None
            else self.config.do_deduplication
        )
        num_doc_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
        )
        # num_beams = num_beams if num_beams is not None else self.config.num_beams
        # NOTE we limit beam search
        num_beams = 1

        assert (
            context_input_ids is not None
        ), "Make sure that context_input_ids is passed!"
        # set to correct device
        context_input_ids = context_input_ids.to(input_ids)

        hypos = []
        kwargs["num_beams"] = num_beams
        kwargs["num_return_sequences"] = num_beams
        kwargs["attention_mask"] = None

        for index in range(len(input_ids)):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[
                index * self.config.n_docs : (index + 1) * self.config.n_docs
            ]  # (n_docs, max_len)

            @profiler("RAGSequence.candidate_generate")
            def candidate_generate():
                candidates = self.generator.generate(
                    generator_input_ids,
                    **kwargs,
                )  # n_docs * n_beam, tgt_len
                return candidates

            output_sequences = candidate_generate()
            if do_deduplication:
                # do_deduplication, max_output_len
                output_sequences = torch.stack(
                    list({str(k.tolist()): k for k in output_sequences}.values())
                )

            # then, run model forwards to get nll scores:
            # new_input_ids = input_ids[index : index + 1].repeat(
            #     len(output_sequences), 1
            # )

            # do tensor reshape
            rag_model_context_input_ids = generator_input_ids.repeat(
                len(output_sequences), 1
            )  # (candidate_size * n_docs, dim)
            rag_model_context_masks = context_masks[
                index * self.config.n_docs : (index + 1) * self.config.n_docs
            ]
            rag_model_context_masks = rag_model_context_masks.repeat(
                len(output_sequences), 1
            )  # (candidate_size * n_docs, dim)
            rag_model_scores = doc_scores[index : (index + 1)]
            rag_model_scores = rag_model_scores.repeat(
                len(output_sequences), 1
            )  # (candidate_size, n_docs)

            # calculate the margin loss
            @profiler("RAGSequence.margin_forward")
            def margin_forward():
                outputs = self(
                    # new_input_ids,
                    context_input_ids=rag_model_context_input_ids,
                    context_attention_mask=rag_model_context_masks,
                    doc_scores=rag_model_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                )
                return outputs

            outputs = margin_forward()
            # choose the best one
            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)
