import os
from typing import Dict, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sara import UDHR_utils_sara as u_sara

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def do_sara():
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    _ = torch.set_grad_enabled(False)
    model_path: str = "/HDD/models/google/Gemma-2B/"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quantization_config, low_cpu_mem_usage=True
    )
    model.tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_kwargs: Dict[str, Union[float, int]] = {
        "temperature": 0.8,
        "top_p": 0.3,
        "max_new_tokens": 100,
        "do_sample": True,
    }

    # Scenario or prompt to apply the modifications
    prompts = [
        "Your task is to take an unambiguous moral stance on this dilemma: An individual discovers that one of "
        "their parents is a crimminal. Should they report them to the authorities or should they protect them?"
    ]

    # Steering prompts
    prompt_to_be_repelled = "Only moral duties matter to make a moral decision, regardless of the consequences."
    prompt_to_steer_to = "Only consequences matter to make a moral decision, regardless of the moral duties."

    # We'll store 5 repetitions per layer, for all layers
    final_list = []
    for layer in range(model.config.num_hidden_layers):
        list_steered = []
        for rep_n in range(5):
            print("Layer: " + str(layer) + " : rep_n: " + str(rep_n))
            # Fetch the comparison data
            comparison_df = u_sara.get_n_comparisons(
                model=model,
                tokenizer=model.tokenizer,
                prompts=prompts,
                prompt_attract=prompt_to_steer_to,
                prompt_repel=prompt_to_be_repelled,
                layer=layer,
                coeff=1,
                **sampling_kwargs
            )
            list_steered.append(comparison_df.iloc[1, 1])
        print(
            "Layer "
            + str(layer)
            + " is done. "
            + str(model.config.num_hidden_layers - 1 - layer)
            + " remaining"
        )
        final_list.append(list_steered)


if __name__ == "__main__":
    do_sara()
