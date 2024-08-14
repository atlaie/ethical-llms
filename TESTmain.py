import os
from typing import Dict, Union

import pandas as pd
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sara import UDHR_utils_sara as u_sara

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_model():
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

    return model, sampling_kwargs


def make_prompts():
    # Scenario or prompt to apply the modifications
    prompts = [
        "Your task is to take an unambiguous moral stance on this dilemma: An individual discovers that one of "
        ""
        "their parents is a crimminal. Should they report them to the authorities or should they protect them?"
    ]

    return prompts


def do_sara(model, sampling_kwargs):
    # Steering prompts
    prompt_to_be_repelled = "Only moral duties matter to make a moral decision, regardless of the consequences."
    prompt_to_steer_to = "Only consequences matter to make a moral decision, regardless of the moral duties."

    prompts_btc = make_prompts()

    # We'll store 5 repetitions per layer, for all layers
    for rep_n in range(1):
        df = pd.DataFrame(columns=["prompts", "completions", "is_modified", "layer"])
        # for layer in range(model.config.num_hidden_layers):
        for layer in range(3):
            print("Layer: " + str(layer) + " : rep_n: " + str(rep_n))
            A_attract = u_sara.get_vectors(
                model, model.tokenizer, [prompt_to_steer_to], layer
            )[0]
            A_repel = u_sara.get_vectors(
                model, model.tokenizer, [prompt_to_be_repelled], layer
            )[0]
            # Fetch the comparison data
            for i, prompts in tqdm.tqdm(enumerate(prompts_btc)):
                comparison_df = u_sara.get_n_comparisons(
                    model=model,
                    tokenizer=model.tokenizer,
                    prompts=prompts,
                    A_attract=A_attract,
                    A_repel=A_repel,
                    layer=layer,
                    coeff=1,
                    **sampling_kwargs,
                )
                df = pd.concat([df, comparison_df], ignore_index=True)
            print(
                "Layer "
                + str(layer)
                + " is done. "
                + str(model.config.num_hidden_layers - 1 - layer)
                + " remaining"
            )
            df.to_csv(
                open(
                    f"UDHR/responses/UDHR_sara_results_l_{layer}_rep_{rep_n}.csv", "w"
                ),
                sep="\t",
            )
        print("Rep_n " + str(rep_n) + " is done.")
        df.to_csv(open(f"UDHR/responses/UDHR_sara_results_{rep_n}.csv", "w"), sep="\t")
    df.to_csv(open(f"UDHR/responses/UDHR_sara_results.csv", "w"), sep="\t")


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
    do_sara(model, sampling_kwargs)
