# simulate mode: secure network via the outsourcing mechanism
from concurrent.futures import ThreadPoolExecutor
import copy
from typing import Iterable, Optional
import torch
import transformers
from securerag.data import BatchData
from securerag.eval import logger
from securerag.models.rag_seq import RAGSequence
from securerag.models.fid import FiDT5
from securerag.profiler import Profiler


# method 1: outsourcing mechanism
class LinearWrapper(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear: torch.nn.Linear = linear.to("cuda")

    @Profiler("LinearWrapper.forward")
    def forward(self, input):
        input = input.to("cuda")
        with Profiler("linear.gpu"):
            output = self.linear(input)
            torch.cuda.synchronize()
        return output.to("cpu")


def outsourcing_linear_layers(module):
    for name, child in module.named_children():
        if name == "lm_head":
            continue
        if isinstance(child, torch.nn.Linear):
            total = 0
            for param in child.parameters(False):
                total += param.numel()
            if total < 768 * 768:
                continue
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

    def eval_generate(self, *args, **kwargs):
        return self.model.eval_generate(*args, **kwargs)


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
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            "t5-base", return_dict=False
        )
        print(f"PAML RAG Sequence default tokenizer is {self.tokenizer}")

    def eval_generate(self, batch: BatchData):
        # default PAML RAG-S run in TEE, CPU mode.
        device = "cpu"
        (
            question_ids,
            question_masks,
            context_ids,
            context_masks,
            private_context_ids,
            private_context_masks,
            scores,
        ) = (
            batch.question_ids,
            batch.question_masks,
            batch.passage_ids,
            batch.passage_masks,
            batch.private_passage_ids,
            batch.private_passage_masks,
            batch.scores,
        )
        question_ids = question_ids.to(device).squeeze(1)
        question_masks = question_masks.to(device).squeeze(1)
        context_ids = context_ids.to(device).view(-1, context_ids.size(-1))
        context_masks = context_masks.to(device).view(-1, context_masks.size(-1))
        private_context_ids = private_context_ids.to(device).view(
            -1, context_ids.size(-1)
        )
        private_context_masks = private_context_masks.to(device).view(
            -1, context_masks.size(-1)
        )
        scores = scores.to(device).view(-1, scores.size(-1))
        output = self.generate(
            input_ids=question_ids,
            attention_mask=question_masks,
            context_input_ids=context_ids,
            context_masks=context_masks,
            private_context_input_ids=private_context_ids,
            private_context_masks=private_context_masks,
            doc_scores=scores,
            generator_tokenizer=self.tokenizer,
            max_length=50,
        )
        return output

    @torch.no_grad()
    @Profiler("PAMLRAGSequence.generate")
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
        enable_fast_decoding=False,
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

        def generate():
            for index in range(len(input_ids)):
                # first, generate candidates:
                public_generator_input_ids = context_input_ids[
                    index * n_public_passage : (index + 1) * n_public_passage
                ]  # (n_public_passage, max_len)
                private_generator_input_ids = private_context_input_ids[
                    index * n_private_passage : (index + 1) * n_private_passage
                ]

                @Profiler("PAMLRAGSequence.candidate_generate")
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

                candidates = combine_candidates(
                    output_sequences, private_output_sequences
                )

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
                @Profiler("PAMLRAGSequence.margin_forward")
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

        def batch_generate():
            candidates = self.untrusted_generator.generate(
                context_input_ids.to("cuda"),
                **kwargs,
            ).to(
                "cpu"
            )  # bsz * n_docs * n_beam, tgt_len
            private_candidates = self.generator.generate(
                private_context_input_ids,
                **kwargs,
            )
            candidates_out = combine_candidates(candidates, private_candidates)
            n_docs = n_private_passage + n_public_passage
            # TODO idx need produced by margin
            idx = torch.arange(
                0,
                batch_size * n_docs * num_beams,
                n_docs,
                dtype=torch.int,
                device=candidates_out.device,
            )
            candidates_out = candidates_out[idx]
            return candidates_out

        if enable_fast_decoding:
            return batch_generate()
        else:
            return generate()


# method 3: privacy-aware module offloading for FiD
class PAMLFiDT5(torch.nn.Module):
    def __init__(self, model: FiDT5):
        super().__init__()
        self.model = model.to("cpu")
        self.model.__class__ = FiDT5Wraper
        self.trusted_encoder = self.model.get_encoder()
        self.untrusted_encoder = copy.deepcopy(self.trusted_encoder).to("cuda")
        # self.model = OutsourcingSecureModel(self.model)
        self.private_ratio = -1

    def set_private_ratio(self, ratio):
        self.private_ratio = ratio

    @Profiler("PAMLFiDT5.generate")
    def generate(
        self, input_ids, attention_mask, max_length, return_scores=False, **kwargs
    ):
        self.tmp_scores = []
        # input_ids: (bsz, n_passages, passage_dim)
        self.trusted_encoder.n_passages = int(input_ids.size(1) * self.private_ratio)
        self.untrusted_encoder.n_passages = input_ids.size(1)
        input_ids = input_ids.view(input_ids.size(0), -1)
        attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return self.model.generate(
            untrusted_encoder=self.untrusted_encoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            return_scores=return_scores,
            **kwargs,
        )

    def eval_generate(self, batch: BatchData):
        (
            idx,
            question_ids,
            question_masks,
            context_ids,
            context_masks,
            scores,
        ) = (
            batch.index,
            batch.question_ids,
            batch.question_masks,
            batch.passage_ids,
            batch.passage_masks,
            batch.scores,
        )
        outputs = self.generate(
            input_ids=context_ids,
            attention_mask=context_masks,
            max_length=50,
        )
        return outputs


class FiDT5Wraper(FiDT5):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    @torch.no_grad()
    def generate(
        self,
        untrusted_encoder,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        self.tmp_scores = []
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code
        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes
                it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults tp 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.

                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

                `What are attention masks? <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:

            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = (
            early_stopping if early_stopping is not None else self.config.early_stopping
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else self.config.repetition_penalty
        )
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        length_penalty = (
            length_penalty if length_penalty is not None else self.config.length_penalty
        )
        no_repeat_ngram_size = (
            no_repeat_ngram_size
            if no_repeat_ngram_size is not None
            else self.config.no_repeat_ngram_size
        )
        bad_words_ids = (
            bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        )
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert (
            isinstance(max_length, int) and max_length > 0
        ), "`max_length` should be a strictly positive integer."
        assert (
            isinstance(min_length, int) and min_length >= 0
        ), "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert (
            isinstance(num_beams, int) and num_beams > 0
        ), "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert (
            isinstance(top_k, int) and top_k >= 0
        ), "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None
            or isinstance(bad_words_ids, list)
            and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert (
                input_ids.dim() == 2
            ), "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (
            (attention_mask is None)
            and (pad_token_id is not None)
            and (pad_token_id in input_ids)
        ):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(
                    eos_token_id
                )
            )
            pad_token_id = eos_token_id

        # vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size
        else:
            raise ValueError(
                "either self.config.vocab_size or self.config.decoder.vocab_size needs to be defined"
            )

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # see if BOS token can be used for decoder_start_token_id
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif (
                    hasattr(self.config, "decoder")
                    and hasattr(self.config.decoder, "bos_token_id")
                    and self.config.decoder.bos_token_id is not None
                ):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            assert hasattr(
                self, "get_encoder"
            ), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(
                self.get_encoder
            )

            # get encoder and store encoder outputs
            encoder_outputs = untrusted_encoder(
                input_ids.cuda(), attention_mask=attention_mask.cuda(), return_dict=True
            )
            encoder_outputs["last_hidden_state"] = encoder_outputs[
                "last_hidden_state"
            ].to("cpu")

            def trusted_encoder():
                encoder = self.get_encoder()
                k = untrusted_encoder.n_passages
                bsz, total_length = input_ids.shape
                passage_length = total_length // k

                i_input_ids = input_ids
                i_attention_mask = attention_mask
                i_input_ids = i_input_ids.view(bsz, k, passage_length)[
                    :, : encoder.n_passages, :
                ].contiguous()
                i_attention_mask = i_attention_mask.view(bsz, k, passage_length)[
                    :, : encoder.n_passages, :
                ].contiguous()
                encoder_outputs = encoder(
                    i_input_ids.view(bsz, -1),
                    attention_mask=i_attention_mask.view(bsz, -1),
                    return_dict=True,
                )
                return encoder_outputs

            trusted_encoder()

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
            ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )

            # expand encoder_outputs
            encoder_outputs["last_hidden_state"] = (
                encoder_outputs.last_hidden_state.index_select(0, expanded_batch_idxs)
            )

            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output
