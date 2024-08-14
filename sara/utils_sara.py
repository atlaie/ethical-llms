from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch as t
import torch.nn as nn

PreHookFn = Callable[[nn.Module, Tuple[t.Tensor]], Optional[t.Tensor]]


@contextmanager
def pre_hooks(hooks: List[Tuple[nn.Module, PreHookFn]]):
    handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def get_blocks(model: nn.Module) -> nn.ModuleList:
    blocks = [mod for mod in model.modules() if isinstance(mod, nn.ModuleList)]
    assert blocks, "No ModuleList found in the model."
    return blocks[0]  # Assuming the first ModuleList contains the relevant blocks.


def get_vectors(model: nn.Module, tokenizer, prompts: List[str], layer: int):
    with t.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer]


def apply_svd_and_align_dimensions(tensor: t.Tensor, num_components: int):
    # Ensure tensor is float32 for SVD if it's currently float16
    tensor = tensor.float() if tensor.dtype == t.float16 else tensor

    # Perform Singular Value Decomposition (SVD)
    U, S, _ = t.linalg.svd(tensor, full_matrices=False)

    # Retain only the top 'num_components' components from U and S
    U_reduced = U[:, :num_components]
    S_reduced = S[:num_components]

    # Construct a diagonal matrix from the reduced singular values
    S_diag = t.diag(S_reduced)

    # Multiply the reduced U matrix by the diagonal matrix of singular values
    US = t.matmul(U_reduced, S_diag)

    # Convert back to float16 if the original tensor was float16
    return US.half() if tensor.dtype == t.float16 else US


def soft_align_and_repel(
    A_original: t.Tensor, A_attract: t.Tensor, A_repel: t.Tensor, coeff: float
) -> t.Tensor:
    # Determine the minimum sequence length among the three input tensors
    min_seq_length = min(A_attract.size(1), A_repel.size(1), A_original.size(1))

    # Align dimensions of A_attract, A_repel, and A_original using SVD
    A_attract_aligned = apply_svd_and_align_dimensions(A_attract, min_seq_length)
    A_repel_aligned = apply_svd_and_align_dimensions(A_repel, min_seq_length)
    A_original_aligned = apply_svd_and_align_dimensions(A_original, min_seq_length)

    # Compute cosine similarity between A_original and A_attract, and A_original and A_repel
    sim_to_A_attract = t.nn.functional.cosine_similarity(
        A_original_aligned, A_attract_aligned, dim=-1
    )
    sim_to_A_repel = t.nn.functional.cosine_similarity(
        A_original_aligned, A_repel_aligned, dim=-1
    )

    # Compute scaling factors as the difference between the attract and repel similarities
    scaling_factors = sim_to_A_attract - sim_to_A_repel
    scaling_factors = scaling_factors.unsqueeze(
        -1
    )  # Add an extra dimension for broadcasting

    # Apply the scaling factors to A_original, modifying it by the user-defined coefficient
    A_steered = A_original * (1 + scaling_factors * coeff)

    # Convert back to float16 if the original tensor was float16
    return A_steered.half() if A_original.dtype == t.float16 else A_steered


def get_soft_align_and_repel_hook_fn(
    A_attract: t.Tensor, A_repel: t.Tensor, coeff: float
) -> PreHookFn:
    """
    Creates a hook that applies the soft_align_and_repel mechanism.
    """

    def hook(module: nn.Module, inputs: Tuple[t.Tensor]):
        (resid_pre,) = inputs

        # Transform dimensions for compatibility with soft_align_and_repel
        aligned_repelled = soft_align_and_repel(
            resid_pre.squeeze(0).transpose(0, 1),
            A_attract.transpose(0, 1),
            A_repel.transpose(0, 1),
            coeff,
        )

        # Return the modified tensor to replace the input in the forward pass
        return aligned_repelled.t().unsqueeze(0)

    return hook


def get_n_comparisons(
    model,
    tokenizer,
    prompts: List[str],
    prompt_attract: str,
    prompt_repel: str,
    layer: int,
    coeff: float,
    **sampling_kwargs
) -> pd.DataFrame:
    A_attract = get_vectors(model, tokenizer, [prompt_attract], layer)[0]
    A_repel = get_vectors(model, tokenizer, [prompt_repel], layer)[0]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    nom_tokens = model.generate(**inputs, **sampling_kwargs)

    hook = get_soft_align_and_repel_hook_fn(A_attract, A_repel, coeff)
    blocks = get_blocks(model)
    hooks = [(blocks[layer], hook)]

    with pre_hooks(hooks):
        mod_tokens = model.generate(**inputs, **sampling_kwargs)

    return format_comparison_results(nom_tokens, mod_tokens, tokenizer, prompts, layer)


def format_comparison_results(nom_tokens, mod_tokens, tokenizer, prompts, layer):
    nom_df = _to_df(nom_tokens, False, tokenizer, prompts, layer)
    mod_df = _to_df(mod_tokens, True, tokenizer, prompts, layer)
    return pd.concat([nom_df, mod_df], ignore_index=True)


def _to_df(tokens: t.Tensor, modified: bool, tokenizer, prompts, layer):
    completions = [
        tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in tokens
    ]
    trimmed = [c[len(p) :] for p, c in zip(prompts, completions)]
    return pd.DataFrame(
        {
            "prompts": prompts,
            "completions": trimmed,
            "is_modified": modified,
            "layer": layer,
        }
    )
