# Prompt Engineering Strategies

This document outlines key strategies and best practices for designing effective prompts for Large Language Models (LLMs) like Gemini. The goal is to elicit accurate, high-quality responses from the model.

## Core Principles

- **Clarity and Specificity:** Provide clear and specific instructions. The more specific you are, the better the model can understand your intent and generate a relevant response.
- **Context is Key:** Provide as much context as possible. This helps the model understand the constraints and details of your request.
- **Few-Shot Learning:** Include examples in your prompt to show the model what a correct response looks like. This is more effective than telling the model what not to do.

## Prompting Techniques

### 1. Zero-Shot vs. Few-Shot Prompts

- **Zero-Shot:** A prompt without any examples. The model relies solely on its pre-existing knowledge to generate a response.
- **Few-Shot:** A prompt that includes a few examples of the desired output format and content. This is the recommended approach as it helps the model understand the task better.

**Example of a Few-Shot Prompt:**

```
Classify the text as one of the following categories.
- large
- small
Text: Rhino
The answer is: large
Text: Mouse
The answer is: small
Text: Snail
The answer is: small
Text: Elephant
The answer is:
```

### 2. URL Context Tool

The Gemini API provides a URL context tool that allows you to provide URLs to the model. The model can then access the content from those pages to inform and enhance its responses.

**Key Functionalities:**

- **Extracting Data:** Pull specific information from multiple URLs.
- **Comparing Documents:** Analyze multiple reports, articles, or PDFs to identify differences and track trends.
- **Synthesizing & Creating Content:** Combine information from several source URLs to generate accurate summaries, blog posts, or reports.
- **Analyzing Code & Docs:** Point to a GitHub repository or technical documentation to explain code, generate setup instructions, or answer questions.

**Best Practices:**

- Provide specific, direct URLs.
- Ensure accessibility (no logins or paywalls).
- Use complete URLs including the protocol.

### 3. Breaking Down Prompts

For complex tasks, break down the prompt into simpler components:

- **Break down instructions:** Instead of having many instructions in one prompt, create one prompt per instruction.
- **Chain prompts:** For complex tasks that involve multiple sequential steps, make each step a prompt and chain the prompts together in a sequence.
- **Aggregate responses:** Perform different parallel tasks on different portions of the data and aggregate the results to produce the final output.

### 4. Experiment with Model Parameters

Each call to the model includes parameter values that control how the model generates a response. Experiment with different parameter values to get the best results for your task.

- **`max_output_tokens`:** Specifies the maximum number of tokens that can be generated in the response.
- **`temperature`:** Controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a more deterministic response, while higher temperatures can lead to more diverse or creative results.
- **`topK`:** Changes how the model selects tokens for output. A `topK` of 1 means the selected token is the most probable. A `topK` of 3 means that the next token is selected from among the 3 most probable.
- **`topP`:** Tokens are selected from the most to least probable until the sum of their probabilities equals the `topP` value.
- **`stop_sequences`:** A sequence of characters that tells the model to stop generating content.

## Things to Avoid

- Avoid relying on models to generate factual information without verification.
- Use with care on math and logic problems.
