<h1 align="center">D-AI-logue synthetic data</h1>

<p align="center">A System for Monitoring and Improving Online Language</p>

Thank you for your interest in our project. Please be aware that this is still a prototipe and may contain bugs or unfinished features.

This repository is part of a project that aims to develop tools that can detect dysfunctinal and toxic language in chat conversations and provide suggestions to make the language more respectful and inclusive.

### Synthetic data

Both `gpt-3.5-turbo` and `dolphin-mistral` were used to generate synthetic data (see files `gen_data_gpt.py`  and `gen_data_dolphin.py`).

The output from the `dolphin-mistral` model was unstable and frequently did not conform to a JSON file format.

To resolve this, I implemented the following:

- Introduced a `system message`: "You are an AI assistant that outputs only JSON data. Do not include any text before or after the JSON response."
- Reiterated the desired format in the `prompt`: "Always respond only with valid JSON format and nothing else. Do not include any text before or after the JSON."

But also with these adjustments, the model now does not generate the desired output format.

So I added in the for loop a way to check if the output is correct and if not it re-run the model.
