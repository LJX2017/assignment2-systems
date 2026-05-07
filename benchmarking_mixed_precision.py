import torch
import torch.nn as nn


# Suppose we are training the model on a GPU and that the model parameters are originally in FP32.
# We’d like to use autocasting mixed precision with FP16. What are the data types of:
# •
# the model parameters within the autocast context?
# •fp32
# the output of the first feed-forward layer (ToyModel.fc1)?
# •fp16
# the output of layer norm (ToyModel.ln)?
# •fp16
# the model’s predicted logits?
# •fp16
# the loss?
# •fp32
# the model’s gradients?
# fp32
def print_type(x):
    print(type(x), x.dtype, x)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print_type(x)
        x = self.relu(self.fc1(x))
        print_type(x)
        x = self.ln(x)
        print_type(x)
        x = self.fc2(x)
        print_type(x)
        return x


model = ToyModel(5, 5)
input = torch.ones(5)
output = torch.ones(5)
print_type(input)
print("begin forwad pass")
dtype: torch.dtype = torch.float16
with torch.autocast(device_type="cpu", dtype=dtype):
    logits = model(input)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(logits, output)
    print("logits type")
    print_type(logits)
    print("loss type")
    print_type(loss)
    loss.backward()
for name, param in model.named_parameters():
    print(name, param.dtype, param.grad.dtype if param.grad is not None else None)
