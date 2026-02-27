import torch

from securerag.utils import add_metric


class Dependency:
    def __init__(self):
        pass

    def resolve(self) -> bool:
        pass


class FusionAggregate(Dependency):
    def __init__(self, public_task, private_task):
        super().__init__()
        self.eta = 2.0
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
        # private_fusion_scores, idx = FusionAggregate.adaptive_passage_selection(
        #     public_scores, private_scores, self.eta
        # )
        private_fusion_contexts = torch.cat([public_contexts, private_contexts], dim=0)
        private_fusion_masks = torch.cat(
            [public_context_masks, private_context_masks], dim=0
        )
        dep_idx = self.get_public_dep_idx()
        private_fusion_scores = torch.cat(
            [public_scores[dep_idx], private_scores], dim=0
        )

        self.private_task.output["contexts"] = private_fusion_contexts
        self.private_task.output["context_masks"] = private_fusion_masks
        self.private_task.input["scores"] = private_fusion_scores
        add_metric("pri_fusion_size", len(private_fusion_scores))
        return True

    def get_public_dep_idx(self):
        public_scores = self.public_task.input["scores"]
        private_scores = self.private_task.input["scores"]
        _, idx = FusionAggregate.adaptive_passage_selection(
            public_scores, private_scores, self.eta
        )
        return idx[: len(idx) - len(private_scores)]

    def adaptive_passage_selection(public_scores, private_scores, eta):
        # all_scores: torch.Tensor = torch.cat([public_scores, private_scores], dim=0)
        # c_size = public_scores.size(0)
        # cp_size = private_scores.size(0)
        # total_size = c_size + cp_size

        # alpha = 1 / (eta * total_size)
        # p_scores = all_scores - torch.full_like(all_scores, alpha)
        # sort_scores, sort_idx = p_scores.sort(descending=True)
        # condition = sort_scores.cumsum(0) - public_scores.sum()
        # if not (condition > 0).any():
        #     return torch.Tensor(), None
        # size = (condition > 0).int().argmax() + 1
        # idx = sort_idx[:size]
        # return all_scores.gather(dim=0, index=idx), idx
        all_scores: torch.Tensor = torch.cat([public_scores, private_scores], dim=0)
        c_size = public_scores.size(0)
        cp_size = private_scores.size(0)
        total_size = c_size + cp_size

        pub_sum = public_scores.sum()
        pri_sum = private_scores.sum()
        all_sum = pub_sum + pri_sum
        threshold = 0.7
        if (pub_sum / all_sum) > threshold or (pri_sum / all_sum) > threshold:
            idx = torch.arange(public_scores.size(0), all_scores.size(0))
        else:
            dist = all_sum * threshold - pri_sum
            pub_sorted, pub_sorted_idx = public_scores.sort(dim=-1, descending=True)
            pub_cumsum = pub_sorted.cumsum(dim=-1)
            first_pos = torch.where(pub_cumsum > dist)[0][0]
            need_size = int(first_pos.item() + 1)
            idx = torch.cat(
                [
                    pub_sorted_idx[:need_size],
                    torch.arange(public_scores.size(0), all_scores.size(0)),
                ],
                dim=0,
            )
        return all_scores[idx], idx


class FinalAggregate(Dependency):
    def __init__(self):
        super().__init__()
        super().__init__()
