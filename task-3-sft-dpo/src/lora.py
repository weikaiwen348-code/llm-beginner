"""手写 LoRA：线性层包装、目标层注入和推理权重合并。"""

import math
import torch
from torch import nn
from pathlib import Path
from transformers import AutoModelForCausalLM

class LoRALinear(nn.Module):
    """保留原线性层，只训练低秩修正分支 A、B。"""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()

        if not isinstance(base, nn.Linear):
            raise TypeError("base 必须是 nn.Linear")
        if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
            raise ValueError("r 必须是正整数")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha 必须是有限正数")

        # 保存原来的层，包括权重和 bias
        self.base = base
        self.base.requires_grad_(False)

        self.scaling = alpha / r

        # 新参数与原权重使用相同的设备和精度
        options = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }

        self.A = nn.Linear(base.in_features, r, bias=False, **options)
        self.B = nn.Linear(r, base.out_features, bias=False, **options)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        original_output = self.base(x)
        small_vector = self.A(x)
        adjustment = self.B(small_vector)
        return original_output + self.scaling * adjustment


def inject_lora(model, target_modules, r, alpha):
    """原地注入 LoRA，返回同一个 model。

    target_modules 可用短名字（如 q_proj），也可用完整模块路径。
    所有基座参数冻结，只有新增 A/B 的参数可以训练。
    """
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    targets = set(target_modules)
    if not targets or any(not isinstance(name, str) or not name for name in targets):
        raise ValueError("target_modules 必须包含非空的模块名字")

    # 先取得快照，再替换模块，避免遍历过程中访问新加入的 A/B。
    modules = list(model.named_modules())
    if any(isinstance(module, LoRALinear) for _, module in modules):
        raise ValueError("模型已经包含 LoRA；请勿重复注入")

    replacements = []
    matched_targets = set()
    for name, module in modules:
        if not isinstance(module, nn.Linear):
            continue
        matches = {
            target for target in targets
            if name == target or name.endswith("." + target)
        }
        if matches:
            matched_targets.update(matches)
            replacements.append((name, module))

    missing = targets - matched_targets
    if missing:
        raise ValueError(f"没有找到目标线性层：{sorted(missing)}")

    # 确认目标都存在，再建立包装层；新 A/B 尚未挂到模型上。
    replacements = [
        (name, LoRALinear(module, r=r, alpha=alpha))
        for name, module in replacements
    ]
    model.requires_grad_(False)
    for name, wrapped in replacements:
        # 如 model.layers.0.self_attn.q_proj：
        # 父路径是 model.layers.0.self_attn，属性名是 q_proj。
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapped)

    return model


@torch.no_grad()
def _merge_linear(layer):
    """合并一个包装层，并返回其原线性层（包括原 bias）。"""
    # PyTorch Linear.weight 的形状是 (out_features, in_features)。
    # B.weight @ A.weight 与它相同；低精度权重用 FP32 计算增量。
    dtype = layer.base.weight.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    delta = (
        layer.B.weight.to(compute_dtype)
        @ layer.A.weight.to(compute_dtype)
    ) * layer.scaling
    layer.base.weight.add_(delta.to(dtype))
    return layer.base


def merge_lora(model):
    """原地合并并移除 LoRA 包装层，返回用于推理的模型。

    使用 model = merge_lora(model)，也支持 model 本身就是 LoRALinear。
    合并后不再保留可继续训练的 A/B，应先保存 adapter 再做合并。
    """
    if isinstance(model, LoRALinear):
        return _merge_linear(model)

    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, _merge_linear(module))
    return model

def get_lora_state(model):
    lora_state = {}
    for name , parameter in model.named_parameters() :
        if parameter.requires_grad == False :
            continue
        lora_state[name] = parameter.detach().cpu().clone()
        
    return lora_state
        
def load_sft_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["lora_config"]
    lora_state = checkpoint["lora_state"]
    
    

    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / config["base_model"]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
    )
    target_modules = config[target_modules]
    r = config[r]
    alpha = config[alpha]

    # TODO：调用 inject_lora，
    # 使用 config 中的 target_modules、r、alpha
    model = inject_lora(model , target_modules , alpha)

    result = model.load_state_dict(lora_state, strict=False)

    missing_lora_keys = [
        key for key in result.missing_keys
        if key.endswith((".A.weight", ".B.weight"))
    ]
    if missing_lora_keys or result.unexpected_keys:
        raise RuntimeError(
            f"LoRA 参数加载失败：缺失参数 {missing_lora_keys}；"
            f"未匹配参数 {result.unexpected_keys}"
        )

    return model
    
    
def _self_test():
    """小模型检查，无需下载或加载 Qwen。"""
    torch.manual_seed(42)
    model = nn.ModuleDict({
        "attention": nn.ModuleDict({
            "q_proj": nn.Linear(16, 8),
            "v_proj": nn.Linear(16, 4),
        }),
        "other": nn.Linear(8, 8),
    })
    x = torch.randn(2, 5, 16)
    before = model["attention"]["q_proj"](x).detach()

    inject_lora(model, target_modules=["q_proj", "v_proj"], r=4, alpha=8)
    layer = model["attention"]["q_proj"]
    assert isinstance(layer, LoRALinear)
    assert torch.allclose(before, layer(x))
    assert layer(x).shape == (2, 5, 8)
    for name, parameter in model.named_parameters():
        expected = name.endswith(".A.weight") or name.endswith(".B.weight")
        assert parameter.requires_grad == expected, name
    print("通过：初始化输出一致，只有 A/B 可训练")

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss = layer(x).square().mean()
    loss.backward()
    assert layer.base.weight.grad is None
    assert layer.B.weight.grad.abs().sum().item() > 0
    optimizer.step()
    assert layer.B.weight.abs().sum().item() > 0
    optimizer.zero_grad(set_to_none=True)
    layer(x).square().mean().backward()
    assert layer.A.weight.grad.abs().sum().item() > 0
    print("通过：基座没有梯度，B 更新后 A 能收到非零梯度")

    # 必须在 B 已非零时测试合并，否则无法发现增量合并错误。
    expected = layer(x).detach()
    model = merge_lora(model)
    actual = model["attention"]["q_proj"](x).detach()
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    model = merge_lora(model)
    assert torch.equal(actual, model["attention"]["q_proj"](x).detach())
    print("通过：非零 LoRA 合并前后输出一致，重复合并不会重复叠加")


if __name__ == "__main__":
    _self_test()
