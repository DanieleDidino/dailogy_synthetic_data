<h1 align="center">Generate Synthetic Data</h1>

Thank you for your interest in our project. Please be aware that this is still a prototype and may contain bugs or unfinished features.

This project aimed at developing tools to detect dysfunctional and toxic language in chat conversations and provide suggestions to make the language more respectful and inclusive.

## Table of Contents

- [Objective](#objective)
- [Large Language Models](#large-language-models)
- [Synthetic Data](#synthetic-data)
- [Model Output](#model-output)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Objective

This repository aims to generate synthetic data to model dysfunctional and toxic language, particularly in interactions between couples or ex-couples who need to communicate regularly. The generated data will be used to develop tools for detecting and mitigating such language, thereby promoting more respectful and inclusive conversations.

## Large Language Models

We employ two advanced Large Language Models (LLMs) for generating synthetic data:

1. `gpt-3.5-turbo` is accessed through the OpenAI API.

2. `dolphin-mistral` from the Ollama Framework:

    - This is a quantized model that runs locally, ensuring privacy and efficiency.
    - The model is based on the Dolphin 2.2.1 version by [Eric Hartford](https://erichartford.com/), which utilizes the Mistral 0.2 framework released in March 2024.
    - It is an uncensored model designed to operate with about 4GB of memory, making it accessible for local deployments.
    - For more information, refer to the [dolphin-mistral model](https://ollama.com/library/dolphin-mistral) and [Dolphin 2.2.1](https://huggingface.co/cognitivecomputations/dolphin-2.2.1-mistral-7b) on Hugging Face.

By utilizing these two LLMs, we generate comprehensive and varied examples of dysfunctional and toxic language, which are crucial for developing tools to detect and mitigate such language in online communications.

## Synthetic Data

The generated examples consist of text reflecting dysfunctional communication, showcasing various forms of toxicity such as insults, harassment, threats, manipulation, and derogatory remarks. We aim to generate sentences that are realistic and diverse in terms of content and context.

The generated text addresses issue categories such as:

- Financial disagreements (e.g., spending habits, debt, child support payments).
- Division of household responsibilities (e.g., chores, maintenance).
- Communication breakdowns (e.g., lack of communication, misunderstandings).
- Differences in parenting styles or decisions.
- Shared custody of children.
- *and so on*

The complete list of issues can be found in the file `utils/issues_category.py`.

## Model Output

Both the `gpt-3.5-turbo` model and `dolphin-mistral` model sometimes do not generate the correct output format (we require a JSON file format). To control this, we implemented a mechanism to check if the output format is correct, and if not, the model re-runs.

## Installation

#### Python version: 3.10.12 

#### 1. Clone the repository

```bash
git clone https://github.com/DanieleDidino/dailogy_synthetic_data.git
```

#### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

#### 3. Activate the virtual environment and install dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

#### 4. Run the script to generate synthetic data
   
```bash
python3 main.py
```

This script will ask a series of questions about the operations to perform.

First, it asks if you want to generate synthetic data (i.e., dysfunctional text) with the OpenAI model.

```bash
Generate syntethic data using gpt-3.5-turbo? (Y/N)
```

This will generate 5 sentences for each issue from the file `utils/issues_category.py`. This number can be changed at the beginning of the script (`N_SENTENCES` parameter).

Second, it asks if you want to generate synthetic data with the model from the Ollama framework.

```bash
Generate syntethic data using dolphin-mistral? (Y/N)
```

Again, this will generate 5 sentences for each issue from the file `utils/issues_category.py` (edit the `N_SENTENCES` parameter to change the number of sentences generated).

Third, it asks if you want to use the OpenAI model to generate a functional version of the dysfunctional text.

```bash
Use gpt-3.5-turbo to generate functional text? (Y/N)
```

The functional text generated in this way is not perfect. Since we aim to use these generated dysfunctional and functional texts as examples in dynamic few-shot prompting, we have manually improved the quality of the text. We edited the generated text to make it more realistic in terms of content and context, similar to the way humans express themselves in everyday life.

Fourth, the script asks if you want to generate the embeddings for the dysfunctional text.

```bash
Generate embeddings with text-embedding-3-small and create SQL database? (Y/N)
```

Normally, one would use something like a PostgreSQL database and store the embeddings as JSONB columns. We will implement this in the future. For now, given the relatively small size (200-300 examples) of our database containing the examples of dysfunctional and functional text, we opted for an SQLite database. It is lightweight and does not require a separate server.

#### 5. Run the script to generate a dynamic few-shot prompt

```bash
python3 generate_dynamic_fewshot_prompt.py
```

This script asks if you want to enter a new text or use a default example (i.e., "*Your poor decisions regarding our child's health show your laziness, putting all the responsibility on me.*")

```bash
You can input a new text or use a defaul example. Input text? (Y/N)
```

This script embeds the new text (or the default example) using `text-embedding-3-small` model and uses cosine similarity to find the N semantically closest examples from the database created above. It then uses the N retrieved examples of dysfunctional-functional language as few-shot examples in the prompt. Examples of the dynamic few-shot prompts generated by this script can be found in the files `dynamic_fewshot_prompts/example_prompt_1.txt` and `dynamic_fewshot_prompts/example_prompt_2.txt`.

We will experiment with this idea in another repository to continuously improve the performance of our app, Dailogy.

#### 6. Deactivate the virtual environment

After generating the synthetic data and the dynamic few-shot prompt, you can deactivate the virtual environment:

```bash
deactivate
```

## Usage

The generated synthetic data can be used for various purposes such as prompt engineering or fine-tuning models to detect and mitigate toxic language, developing chatbots that can handle difficult conversations more empathetically, and improving online communication tools.

## Contributing

We welcome contributions to this project! If you have suggestions for improvements or have found a bug, please feel free to contact us.

## Contact

For questions or further information, please contact [Daniele Didino](https://www.linkedin.com/in/daniele-didino).
