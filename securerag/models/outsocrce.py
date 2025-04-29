# simulate mode: secure network via the outsocring mechanism
import torch


class LinearWrapper(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear: torch.nn.Linear = linear.to("cuda")

    def forward(self, input):
        # linear size 
        input = input.to("cuda")
        output = self.linear.forward(input)
        return output.to("cpu")


def outsocring_linear_layers(module):
    for name, child in module.named_children():
        if name == "lm_head":
            continue
        if isinstance(child, torch.nn.Linear):
            linear_wrapper = LinearWrapper(child)
            setattr(module, name, linear_wrapper)
        else:
            outsocring_linear_layers(child)


class OutsocringSecureModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.to("cpu")
        outsocring_linear_layers(self.model)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
