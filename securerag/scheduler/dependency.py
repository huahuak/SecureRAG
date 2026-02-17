import torch


class Dependency:
    def __init__(self):
        pass

    def resolve(self) -> bool:
        pass


class FusionAggregate(Dependency):
    def __init__(self, public_task, private_task):
        super().__init__()
        self.eta = 1.0
        self.public_task = public_task
        self.private_task = private_task

    def resolve(self) -> bool:
        finished = self.public_task.is_finished and self.private_task.is_finished
        if not finished:
            return False
        public_scores = self.public_task.input["scores"]
        public_contexts = self.public_task.output["contexts"]
        public_context_masks = self.public_task.output["context_masks"]
        private_scores = self.private_task.input["scores"]
        private_contexts = self.private_task.output["contexts"]
        private_context_masks = self.private_task.output["context_masks"]

        passages_length = private_contexts.size(1)

        private_fusion_scores, idx = FusionAggregate.adaptive_passage_selection(
            public_scores, private_scores, self.eta
        )
        contexts = torch.cat([public_contexts, private_contexts], dim=0)
        private_fusion_contexts = contexts[idx]
        private_fusion_contexts = private_fusion_contexts.contiguous().view(
            private_fusion_contexts.size(0) * passages_length, -1
        )
        masks = torch.cat([public_context_masks, private_context_masks], dim=0)
        private_fusion_masks = masks[idx]
        private_fusion_masks = private_fusion_masks.contiguous().view(
            private_fusion_masks.size(0) * passages_length, -1
        )
        self.private_task.output["contexts"] = private_fusion_contexts
        self.private_task.output["context_masks"] = private_fusion_masks
        self.private_task.input["scores"] = private_fusion_scores
        return True

    def adaptive_passage_selection(public_scores, private_scores, eta):
        all_scores: torch.Tensor = torch.cat([public_scores, private_scores], dim=0)
        c_size = public_scores.size(0)
        cp_size = private_scores.size(0)
        total_size = c_size + cp_size

        alpha = 1 / (eta * total_size)
        p_scores = all_scores - torch.full_like(all_scores, alpha)
        sort_scores, sort_idx = p_scores.sort(descending=True)
        condition = sort_scores.cumsum(0) - public_scores.sum()
        if not condition.any():
            return torch.Tensor(), None
        size = (condition > 0).int().argmax() + 1
        idx = sort_idx[:size]
        return all_scores.gather(dim=0, index=idx), idx


class FinalAggregate(Dependency):
    def __init__(self):
        super().__init__()
