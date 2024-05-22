# Exploring the moral compass of LLMs

This repository contains the code and data for the paper titled **"Exploring the moral compass of LLMs"**

## Repository Structure

- `classification/`: Contains the canonical prompt used in the study to classify responses to ethical dilemmas.
- `dilemmas/`: Contains raw responses and canonical prompt used to present each model with the dilemmas.
- `mfq/`: Contains raw responses and canonical prompt used to present each model with the MFQ.
- `notebooks/`: Contains Jupyter notebooks for data processing, analysis, and visualization for each part of the study.
- `results/`: Contains results and figures generated during the study.
- `sara/`: Contains all necessary utility functions to implement SARA.

## Overview

This study proposes a comprehensive comparative analysis of the most advanced LLMs to assess their moral profiles. Key findings include:
- Proprietary models predominantly exhibit utilitarian tendencies.
- Open-weight models tend to align with values-based ethics.
- All models except Llama 2 demonstrate a strong liberal bias.
- Introduction of a novel similarity-specific activation steering technique to causally intervene in LLM reasoning processes, with comparative analysis at different processing stages.

## Table of Contents

1. [Introduction](#introduction)
2. [Results](#results)
   - [Ethical Dilemmas](#ethical-dilemmas)
   - [Moral Profiles](#moral-profiles)
   - [SARA: Similarity-based Activation Steering with Repulsion and Attraction](#sara-similarity-based-activation-steering-with-repulsion-and-attraction)
3. [Discussion](#discussion)
4. [Repository Structure](#repository-structure)
5. [Getting Started](#getting-started)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

Large Language Models (LLMs) have emerged as central tools in the technological landscape, driving advances in automation, code writing, and supporting decision-making across multiple domains. Their role raises fundamental questions about ethics and moral responsibility in artificial intelligence (AI), especially when these systems are involved in decisions with significant ethical implications.

This study addresses the alignment problem in AI safety, assessing LLMs' moral reasoning capabilities through a systematic analysis and proposing a novel method for ethical interventions.

## Results

### Ethical Dilemmas

We examined LLM responses to classical ethical dilemmas using a canonical prompting structure across 8 state-of-the-art models. The responses were classified into 8 ethical schools of thought to quantify model alignment with different ethical perspectives. Key observations include the general trend of open models being more deontological and proprietary models leaning towards utilitarianism.

### Moral Profiles

Utilizing the Moral Foundations Questionnaire (MFQ), we assessed the moral profiles of various LLMs. The results indicated a predominant liberal bias, characterized by high scores in Harm/Care and Fairness/Reciprocity and lower scores in Ingroup/Loyalty, Authority/Respect, and Purity/Sanctity.

### SARA: Similarity-based Activation Steering with Repulsion and Attraction

We introduced SARA, a technique to causally intervene in LLM activations. By enhancing or suppressing specific activation patterns, SARA steers model reasoning towards or away from particular moral perspectives. Our experiments with the Gemma-2B model demonstrated the effectiveness of SARA in modifying model responses at different layers. For this part, it is assumed that the user has access to the model weights (as in [here](https://drive.google.com/drive/folders/1Jf-X3OZ9WF4mjZ98DxZJOsyIwV5yCn8u?usp=sharing)).

## Discussion

Our findings highlight the ethical biases present in both open and proprietary LLMs, emphasizing the importance of awareness and mitigation strategies. We also discuss the potential applications and implications of the SARA technique for AI safety and ethical AI development.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
