# chatbot
This is an interactive chatbot system that supports end-to-end traffic prediction tasks through natural language. It integrates powerful large language models (LLMs) including GPT-3.5 and DeepSeek to enable a fully automated forecasting workflow.

# Key Features

**1.LLM-driven Natural Language Forecasting**
Users can describe their intent freely (e.g. “How is the traffic for IP 10196 next day?”), and the system will parse parameters, infer context, and execute predictions accordingly.

**2.Multi-model Support**
Supports both OpenAI GPT and locally deployed DeepSeek LLMs. Backend is model-agnostic, allowing easy switching and cost-accuracy balancing.

**3.Multi-round Clarification with Context**
When user input is ambiguous, the system uses GPT to ask follow-up questions. It also maintains conversation history to support continuation commands like “same time next day”.

**4.Time-series Forecasting (TFT-based)**
Uses a pre-trained Temporal Fusion Transformer model to perform accurate network traffic prediction over multiple time scales.

**5.Automatic Result Explanation**
Generates text-based interpretation of model outputs, including detection of anomalies, risk alerts, and explanation of accuracy.

**6.Visual Output + Captioning (In Progress)**
Predictive charts are generated for each task, and LLMs generate image captions to summarize key trends (code supported; runtime still in debugging phase).

# Use Case
Users can input queries like:

“Predict the traffic of the second day after the last IP”

“I'd like to check the traffic of IP 10196 in CESNET at 10:00 on June 3, 2024”

“Let's switch to another IP and take a look at the traffic trend of last week”

The system will intelligently understand the task, extract parameters, predict results using TFT, and explain them through LLM-generated summaries.


# Model: Temporal Fusion Transformer (TFT)
This project uses the Temporal Fusion Transformer for time series forecasting. The model is trained to predict future traffic values (in Mbps or Bytes) given historical sequences. It supports:

Multi-horizon prediction；

Variable selection and attention mechanisms；

Uncertainty estimation (optionally)；

The prediction process outputs both metrics (e.g., SMAPE, R², correlation) and visualizations.

# Datasets
**CIC-IDS2018:** Public intrusion detection dataset including attack traffic.

**CESNET:** Real-world backbone traffic collected by CESNET (IP-level).

Each dataset is loaded through custom adapters that support flexible extraction and preprocessing for model consumption.

# LLM Integration
**OpenAI GPT-3.5**: Used for complex natural language understanding and context-sensitive clarification.

**DeepSeek-V2-Chat:** Local model (7B or smaller) for efficient lightweight processing (parameter parsing, summarization).

Prompt-driven pipeline with examples in /prompts/.

# Auto Explanation & Image Captioning
After each prediction:

Graphs (matplotlib) are generated for actual vs. predicted traffic.

LLMs generate concise captions and summaries for user interpretation.

Supports summarization across multiple predictions (trend extraction).


If you're interested in the project or would like to collaborate, feel free to reach out!
