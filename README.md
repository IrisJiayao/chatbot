# chatbot
This is an interactive chatbot system that supports end-to-end traffic prediction tasks through natural language. It integrates powerful large language models (LLMs) including GPT-3.5 and DeepSeek to enable a fully automated forecasting workflow.
Key Features
LLM-driven Natural Language Forecasting
Users can describe their intent freely (e.g. “How is the traffic for IP 10196 next day?”), and the system will parse parameters, infer context, and execute predictions accordingly.

Multi-model Support
Supports both OpenAI GPT and locally deployed DeepSeek LLMs. Backend is model-agnostic, allowing easy switching and cost-accuracy balancing.

Multi-round Clarification with Context
When user input is ambiguous, the system uses GPT to ask follow-up questions. It also maintains conversation history to support continuation commands like “same time next day”.

Time-series Forecasting (TFT-based)
Uses a pre-trained Temporal Fusion Transformer model to perform accurate network traffic prediction over multiple time scales.

Automatic Result Explanation
Generates text-based interpretation of model outputs, including detection of anomalies, risk alerts, and explanation of accuracy.

Visual Output + Captioning (In Progress)
Predictive charts are generated for each task, and LLMs generate image captions to summarize key trends (code supported; runtime still in debugging phase).
