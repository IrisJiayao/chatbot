# server.py
#
# This code implements a multi-step dialogue flow for a socket-based "Oracle" server.
# The steps are:
#   Step 0: Choose dataset (CIC-IDS2018 or CESNET)
#   Step 1: Choose single-step or multistep
#   Step 2: Choose analysis type (overall / timeslot / top10)
#       - If dataset = CIC, top10 is not allowed
#       - If timeslot => jump to step=4 to parse user time
#       - If top10 => jump to step=5 with immediate result
#       - Otherwise => step=3 if needed (CESNET, "network or ip")
#   Step 3: For CESNET's overall/timeslot => ask "network" or "ip"
#   Step 4: If "ip", user just enters a code like "100610", or if "timeslot", user enters a single time point.
#   Step 5: final results or wait 'restart'/'exit'

import socket
import threading
import nltk
import string
import re
import os
import json
from pathlib import Path
from datetime import datetime
import time
from llm_utils import call_llm, load_prompt_template, call_llm_conversational, finalize_prediction_parameters

# Import the adapters
from CICInferenceAdapter import CICAdapter
from CESNETInferenceAdapter import CESNETAdapter

# GPT 接口调用函数
import openai

#openai.api_key = "----"  # 替换成你自己的 API key

client = openai.OpenAI(api_key="------")  # 用你的 Key，注意是 `OpenAI()` 而不是 `openai.api_key =`
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"  # 可改为 "deepseek"

# 初始化 session_state
state = {
    "step": 0,
    "dataset": None,
    "analysis_step": None,
    "analysis_type": None,
    "target_scope": None,
    "target_ip": None,
    "time_point": None,
    "history": [],  # 新增：对话历史
    # 在初始化 session_state 时加：
    "last_prediction_context": {
        "dataset": None,
        "analysis_type": None,
        "analysis_step": None,
        "target_scope": None,
        "target_ip": None,
        "time_point": None
    }
}

def update_prediction_context(session_state):
    session_state["last_prediction_context"] = {
        "dataset": session_state["dataset"],
        "analysis_type": session_state["analysis_type"],
        "analysis_step": session_state["analysis_step"],
        "target_scope": session_state["target_scope"],
        "target_ip": session_state["target_ip"],
        "time_point": session_state["time_point"]
    }



def call_gpt(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 或 gpt-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant that helps users configure forecasting tasks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT ERROR] {e}")
        return "⚠️ GPT 出错了，无法处理请求。"
import json

# def extract_parameters_from_nl(user_query: str):
#     prompt = f"""
# You are an AI assistant helping extract structured configuration from user instructions for a network forecasting system.
#
# Please extract the following fields from the user's message:
# - dataset: 'CIC-IDS2018' or 'CESNET'
# - analysis_type: 'overall', 'timeslot', or 'top10'
# - target_ip: string (only for CESNET & if applicable)
# - time_point: time in format YYYY-MM-DD-HH:MM (only if timeslot)
#
# Respond in valid JSON only. If a field is not specified, set it to null.
#
# User input:
# \"\"\"{user_query}\"\"\"
# """
#
#     try:
#         response = call_gpt(prompt)
#         return json.loads(response)
#     except Exception as e:
#         print(f"[GPT PARAM ERROR] {e}")
#         return None
def build_prediction_intro(session_state):
    dataset = session_state.get("dataset", "Unknown")
    atype = session_state.get("analysis_type", "Unknown")
    ip = session_state.get("target_ip")
    tpoint = session_state.get("time_point")

    desc = f"Forecasting traffic using {dataset}"
    if ip:
        desc += f", target IP: {ip}"
    if tpoint:
        desc += f", time: {tpoint}"
    desc += f", type: {atype}\nPlease wait...\n"

    return desc

# def extract_parameters_from_nl(user_query: str, model_name=DEFAULT_LLM_MODEL):
#     prompt = load_prompt_template("extract_parameters", {"USER_INPUT": user_query})
#     try:
#         response = call_llm(model_name, prompt)
#         return json.loads(response)
#     except Exception as e:
#         print(f"[PARAM PARSE ERROR] {e}")
#         return None


from datetime import datetime, timedelta

def is_vague_query(user_input: str, model_name="gpt"):
    prompt = f"""
You are a system that determines whether the user's instruction is **vague** (referring to previous context).

If the instruction refers to **something already said**, like "same time", "last IP", "predict again", "another one", "same conditions", etc., then it's vague.

User input: "{user_input}"

Respond strictly with one word: **"yes"** or **"no"**.
"""
    reply = call_llm(prompt, model=model_name).strip().lower()
    return reply.startswith("yes")


def extract_parameters_from_nl(user_query: str, model_name=DEFAULT_LLM_MODEL, ctx=None):
    user_query_lower = user_query.lower()

    # 🧠 判断是否为“模糊表达”
    use_context = False
    if ctx:
        use_context = is_vague_query(user_query, model_name)

    # ✂️ 构造上下文 block（仅在需要时）
    ctx_block = ""
    if use_context and ctx:
        lines = []
        if ctx.get("dataset"):
            lines.append(f"- Dataset: {ctx['dataset']}")
        if ctx.get("analysis_type"):
            lines.append(f"- Analysis Type: {ctx['analysis_type']}")
        if ctx.get("target_ip"):
            lines.append(f"- Target IP: {ctx['target_ip']}")
        if ctx.get("time_point"):
            lines.append(f"- Time Point: {ctx['time_point']}")
        if lines:
            ctx_block = "Previous context:\n" + "\n".join(lines)

    # 📜 拼接 Prompt
    base_prompt = load_prompt_template("extract_parameters", {
        "USER_INPUT": user_query
    })
    full_prompt = f"{base_prompt}\n\n{ctx_block}" if ctx_block else base_prompt

    # 🧪 调用模型提取参数
    try:
        response = call_llm(full_prompt, model=model_name)
        return json.loads(response)
    except Exception as e:
        print(f"[PARAM PARSE ERROR] {e}")
        return None



def generate_summary(result: dict, session_state: dict, model_name=DEFAULT_LLM_MODEL):
    prompt = load_prompt_template("summarize_prediction", {
        "DATASET": session_state["dataset"],
        "ANALYSIS_TYPE": session_state["analysis_type"],
        "SCOPE": session_state["target_ip"] if session_state["target_scope"] == "ip" else session_state["target_scope"],
        "TIME_POINT": session_state.get("time_point", "N/A"),
        "PREDICTION": result.get("prediction", "N/A"),
        "ACTUAL": result.get("actual", "N/A"),
        "ACCURACY": result.get("accuracy", "N/A")
    })
    return call_llm(prompt, model=model_name)


def generate_explanation(result: dict, session_state: dict, model_name=DEFAULT_LLM_MODEL):
    """
    使用 prompt 模板解释预测结果的原因（适用于流量高/低或准确率低时）
    """
    prompt = load_prompt_template("explain_traffic", {
        "DATASET": session_state["dataset"],
        "SCOPE": session_state["target_ip"] if session_state["target_scope"] == "ip" else session_state["target_scope"],
        "TIME_POINT": session_state.get("time_point", "N/A"),
        "PREDICTION": result.get("prediction", "N/A"),
        "ACTUAL": result.get("actual", "N/A"),
        "ACCURACY": result.get("accuracy", "N/A")
    })
    return call_llm(prompt, model=model_name)

def clarify_user_intent(user_input: str, model_name=DEFAULT_LLM_MODEL):
    """
    当用户输入模糊不清时，使用 prompt 引导用户明确意图
    """
    prompt = load_prompt_template("clarify_intent", {
        "USER_INPUT": user_input
    })
    return call_llm(prompt, model=model_name)



# If not downloaded:
# nltk.download('punkt')
# nltk.download('stopwords')

try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except:
    # Fallback if NLTK data is not available
    stop_words = set(
        ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
         'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
         'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
         'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
         'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
         'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
         'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
         'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
         'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
         'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
         "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
         'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
         'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
         'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
puncts = set(string.punctuation)

# Configuration paths
cesnet_config_path = "D:\\PythonProject\\chatbot\\config\\config\\CESNET.yaml"
cic_config_path = "D:\\PythonProject\\chatbot\\config\\config\\CSE-CIC-IDS2018.yaml"

# Initialize the CESNET adapter
try:
    cesnet_adapter = CESNETAdapter(cesnet_config_path)
    print("✅ CESNET adapter initialized successfully")
except Exception as e:
    print(f"⚠️ Error initializing CESNET adapter: {e}")
    cesnet_adapter = None

# Initialize the CIC-IDS2018 adapter
try:
    cic_adapter = CICAdapter(cic_config_path)
    print("✅ CIC-IDS2018 adapter initialized successfully")
except Exception as e:
    print(f"⚠️ Error initializing CIC-IDS2018 adapter: {e}")
    cic_adapter = None


# Removed fake_traffic_predict as the adapters have built-in fallback mechanisms




# def format_prediction_response(result, session_state):
#     """
#     Format the prediction result into a user-friendly response.
#
#     Args:
#         result (dict): Prediction result from the adapter
#         session_state (dict): Current session state
#
#     Returns:
#         str: Formatted response message
#     """
#     if not result["success"]:
#         return f"Error: {result['message']}\nType 'restart' to try again or 'exit' to quit.\n"
#
#     # Extract basic info
#     dataset = session_state["dataset"]
#     analysis_type = session_state["analysis_type"]
#     target_scope = session_state["target_scope"]
#     target_ip = session_state["target_ip"]
#     time_point = session_state["time_point"]
#     is_multistep = session_state["analysis_step"] == "multistep"
#
#     # Start building the response
#     if dataset == "CESNET":
#         header = f"[CESNET"
#     else:
#         header = f"[CIC-IDS2018"
#
#     # Add analysis type
#     if analysis_type == "overall":
#         header += " - Overall"
#     elif analysis_type == "timeslot":
#         header += f" - Timeslot {time_point}"
#     elif analysis_type == "top10":
#         header += " - Top10 IPs"
#
#     # Add scope
#     if target_scope == "network":
#         header += " Network"
#     elif target_scope == "ip" and target_ip:
#         ip_desc = result.get("details", {}).get("ip_description", "")
#         header += f" IP {target_ip}"
#         if ip_desc:
#             header += f" ({ip_desc})"
#
#     header += "]"
#
#     # Build the prediction part
#     # Ensure prediction is in proper units (Mbps or Bytes)
#     if dataset == "CESNET":
#         # CESNET returns Bytes
#         prediction = f"Predicted traffic: {result['prediction']} Bytes"
#     else:
#         # CIC returns Mbps
#         prediction = f"Predicted traffic: {result['prediction']} Mbps"
#
#     if result.get("accuracy") is not None:
#         prediction += f" (Accuracy: {result['accuracy']}%)"
#
#     # Add details for top10 if applicable
#     details = ""
#     if analysis_type == "top10" and "top_ips" in result.get("details", {}):
#         details += "\n\nTop IPs by traffic volume:"
#         top_ips = result["details"]["top_ips"]
#         for i, (ip, data) in enumerate(sorted(top_ips.items(), key=lambda x: x[1]["traffic"], reverse=True), 1):
#             desc = data.get("description", "")
#             details += f"\n{i}. IP {ip}: {data['traffic']:.2f} Bytes" if dataset == "CESNET" else f"\n{i}. IP {ip}: {data['traffic']:.2f} Mbps"
#             if desc:
#                 details += f" ({desc})"
#
#     # Add graph information if available
#     graphs = ""
#     if result.get("graphs"):
#         graphs += "\n\nGraphs generated:"
#         for i, graph_path in enumerate(result["graphs"], 1):
#             graphs += f"\n{i}. {os.path.basename(graph_path)}"
#
#     # Combine all parts
#     response = f"{header}\n{prediction}{details}{graphs}\n\nType 'restart' to predict again or 'exit' to quit.\n"
#
#     return response

def format_prediction_response(result, session_state):
    """
    Format the prediction result into a user-friendly response.

    Args:
        result (dict): Prediction result from the adapter
        session_state (dict): Current session state

    Returns:
        str: Formatted response message
    """
    if not result["success"]:
        return f"Error: {result['message']}\nType 'restart' to try again or 'exit' to quit.\n"

    # Extract basic info
    dataset = session_state["dataset"]
    analysis_type = session_state["analysis_type"]
    target_scope = session_state["target_scope"]
    target_ip = session_state["target_ip"]
    time_point = session_state["time_point"]
    is_multistep = session_state["analysis_step"] == "multistep"

    # Start building the response
    if dataset == "CESNET":
        header = f"[CESNET"
    else:
        header = f"[CIC-IDS2018"

    # Add analysis type
    if analysis_type == "overall":
        header += " - Overall"
    elif analysis_type == "timeslot":
        header += f" - Timeslot {time_point}"
    elif analysis_type == "top10":
        header += " - Top10 IPs"

    # Add scope
    if target_scope == "network":
        header += " Network"
    elif target_scope == "ip" and target_ip:
        ip_desc = result.get("details", {}).get("ip_description", "")
        header += f" IP {target_ip}"
        if ip_desc:
            header += f" ({ip_desc})"

    header += "]"

    # Build the prediction part
    # Ensure prediction is in proper units (Mbps or Bytes)
    if dataset == "CESNET":
        # CESNET returns Bytes
        prediction = f"Predicted traffic: {result['prediction']} Bytes"
    else:
        # CIC returns Mbps
        prediction = f"Predicted traffic: {result['prediction']} Mbps"

    if result.get("accuracy") is not None:
        prediction += f" (Accuracy: {result['accuracy']}%)"

    # Add details for top10 if applicable
    details = ""
    if analysis_type == "top10" and "top_ips" in result.get("details", {}):
        details += "\n\nTop IPs by traffic volume:"
        top_ips = result["details"]["top_ips"]
        for i, (ip, data) in enumerate(sorted(top_ips.items(), key=lambda x: x[1]["traffic"], reverse=True), 1):
            desc = data.get("description", "")
            details += f"\n{i}. IP {ip}: {data['traffic']:.2f} Bytes" if dataset == "CESNET" else f"\n{i}. IP {ip}: {data['traffic']:.2f} Mbps"
            if desc:
                details += f" ({desc})"

    # Add graph information if available
    graphs = ""
    if result.get("graphs"):
        graphs += "\n\nGraphs generated:"
        for i, graph_path in enumerate(result["graphs"], 1):
            graphs += f"\n{i}. {os.path.basename(graph_path)}"

    # === 自动图表 caption 调用（if single image） ===
    caption = ""
    if result.get("graphs") and len(result["graphs"]) == 1 and target_ip and time_point:
        try:
            from llm_utils import generate_caption_from_chart
            caption_text = generate_caption_from_chart(dataset, target_ip, time_point)
            caption = f"\n\nCaption: {caption_text}"
        except Exception as e:
            caption = f"\n\nCaption generation failed: {e}"

    response = f"{header}\n{prediction}{details}{graphs}{caption}\n\nType 'restart' to predict again or 'exit' to quit.\n"

    return response



# Session management
session_states = {}


def get_session_state(addr):
    """Retrieve or create a session state dictionary for a given client address."""
    if addr not in session_states:
        session_states[addr] = {
            "step": 0,
            "dataset": None,  # "CIC-IDS2018" or "CESNET"
            "analysis_step": None,  # "single-step" or "multistep"
            "analysis_type": None,  # "overall", "timeslot", "top10"
            "time_point": None,  # string like "2025-05-04-10:00"
            "target_scope": None,  # "network" or "ip"
            "target_ip": None  # user input for IP code, e.g. "100610"
        }
    return session_states[addr]


def reset_session_state(addr):
    """Remove the session state for a client."""
    if addr in session_states:
        session_states.pop(addr)


def preprocess_text(text: str):
    """Lowercase, tokenize, remove stopwords/punct."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    return [t for t in tokens if t not in stop_words and t not in puncts]


def parse_single_time_point(text: str):
    """
    Check if text matches YYYY-MM-DD-HH:MM, e.g. '2025-05-04-10:00'.
    Returns the match or None if not found.
    """
    pattern = r"\b(\d{4})-(\d{2})-(\d{2})-(\d{2}):(\d{2})\b"
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None


def make_prediction(session_state):
    """
    Make a prediction using the appropriate adapter based on the session state.

    Args:
        session_state (dict): Current session state

    Returns:
        dict: Prediction result
    """
    dataset = session_state["dataset"]
    analysis_type = session_state["analysis_type"]
    target_scope = session_state["target_scope"]
    target_ip = session_state["target_ip"]
    time_point = session_state["time_point"]
    is_multistep = session_state["analysis_step"] == "multistep"

    # Select the appropriate adapter
    if dataset == "CESNET":
        adapter = cesnet_adapter
    else:  # CIC-IDS2018
        adapter = cic_adapter

    # Make the prediction using the adapter
    try:
        result = adapter.predict(
            analysis_type=analysis_type,
            target_scope=target_scope if dataset == "CESNET" else "network",
            target_ip=target_ip if dataset == "CESNET" else None,
            time_point=time_point if analysis_type == "timeslot" else None,
            is_multistep=is_multistep
        )

        return result
    except Exception as e:
        print(f"Error making prediction with adapter: {e}")
        # Return a simple error response
        return {
            "success": False,
            "message": f"Error making prediction: {str(e)}",
            "details": {}
        }


# Removed generate_fake_prediction as the adapters have built-in fallback mechanisms


def conversation_flow(user_query: str, session_state: dict):
    """
    Main state machine for multi-step dialogue.

    Args:
        user_query (str): User's input
        session_state (dict): Current session state

    Returns:
        tuple: (response_message, is_prediction_result)
    """
    step = session_state["step"]
    dataset = session_state["dataset"]
    analysis_step = session_state["analysis_step"]
    analysis_type = session_state["analysis_type"]
    time_point = session_state["time_point"]
    target_scope = session_state["target_scope"]
    target_ip = session_state["target_ip"]

    user_lower = user_query.lower().strip()

    # 特殊触发词：如果用户输入太自然语言，就试图解析所有参数直接预测
    if session_state["step"] == 0 and len(user_query.split()) > 5:
        # extracted = extract_parameters_from_nl(user_query)

        ctx = session_state.get("last_prediction_context") or {}
        extracted = extract_parameters_from_nl(user_query, ctx=ctx)

        # 判断是否缺少必要字段，才用 context 补齐
        required_fields = ["dataset", "analysis_type"]
        if not extracted or any(k not in extracted or extracted[k] is None for k in required_fields):
            # 用 context 补全（如果有）
            ctx = session_state.get("last_prediction_context") or {}
            extracted = finalize_prediction_parameters(extracted or {}, ctx)

        if extracted:

            # 自动填入 session_state
            dataset = extracted.get("dataset")
            atype = extracted.get("analysis_type")
            ip = extracted.get("target_ip")
            tpoint = extracted.get("time_point")

            if dataset in ["CIC-IDS2018", "CESNET"] and atype:
                session_state["dataset"] = dataset
                session_state["analysis_type"] = atype
                session_state["analysis_step"] = "single-step"  # 默认即可
                session_state["target_scope"] = "network" if not ip else "ip"
                session_state["target_ip"] = ip
                session_state["time_point"] = tpoint
                session_state["step"] = 5

                # intro_msg = build_prediction_intro(session_state)
                # print(intro_msg)
                # result = make_prediction(session_state)
                # return (intro_msg + format_prediction_response(result, session_state), True)
                intro_msg = build_prediction_intro(session_state)
                return (intro_msg, "DEFERRED_PREDICTION")

        # # 如果提取失败，说明用户表达不清楚：调用 GPT 来引导
        # clarification = clarify_user_intent(user_query, model_name=DEFAULT_LLM_MODEL)
        # return (clarification, False)

        # 参数不全或无法提取，先试试 GPT 静态澄清
        clarification = clarify_user_intent(user_query, model_name=DEFAULT_LLM_MODEL)
        if clarification:
            return (clarification, False)

        # 最后兜底：调用上下文感知 GPT 多轮澄清
        gpt_reply = call_llm_conversational(user_query, session_state, model="gpt")
        return (
            f"I couldn't fully understand your request.\nGPT suggests: **{gpt_reply}**\n\nYou can type it directly or rephrase.\n",
            False
        )


    # ------------------- STEP 0: Choose dataset -------------------
    if step == 0:
        if "cic" in user_lower:
            session_state["dataset"] = "CIC-IDS2018"
            session_state["step"] = 1
            return (
                "You chose CIC-IDS2018. Do you want 'single-step' or 'multistep'? (Multistep can predict network traffic over a larger time horizon but take longer to analyse; single-step predict a smaller time horizon but take less time to analyse.)\n",
                False)
        elif "cesnet" in user_lower:
            session_state["dataset"] = "CESNET"
            session_state["step"] = 1
            return (
                "You chose CESNET. Do you want 'single-step' or 'multistep'? (Multistep can predict network traffic over a larger time horizon but take longer to analyse; single-step predict a smaller time horizon but take less time to analyse.)\n",
                False)
        # else:
        #     # ✅ 新增 GPT 助力推断
        #     # gpt_reply = call_gpt(
        #     #     f"The user said: '{user_query}'. Please decide which dataset they want: 'CIC-IDS2018' or 'CESNET'. Reply with only one of the two options.")
        #     gpt_reply = call_gpt(f"""
        #     You are helping a user choose between two network traffic datasets: 'CIC-IDS2018' and 'CESNET'.
        #
        #     Here are the descriptions:
        #     - CIC-IDS2018: Simulated dataset with aggregated network-level traffic, no specific IP information.
        #     - CESNET: Real-world dataset with detailed IP-level traffic and timestamps.
        #
        #     The user said: "{user_query}"
        #
        #     Based on their intent, decide which dataset they are referring to.
        #     Reply with only one word: 'CIC-IDS2018' or 'CESNET'.
        #     """)
        #
        #     return (
        #     f"Your input was unclear.\nGPT suggests you meant: {gpt_reply}\nIf this is correct, type it. Or type 'CIC-IDS2018' or 'CESNET' explicitly.\n",
        #     False)

        #
        else:
            # 2. 尝试 LLM 静态澄清
            guess = clarify_user_intent(user_query)
            if guess in ["CIC-IDS2018", "CESNET"]:
                return (f"I think you meant: **{guess}**\nType it to confirm.\n", False)

            # 3. 最终兜底：走多轮上下文澄清
            gpt_reply = call_llm_conversational(user_query, session_state, model=DEFAULT_LLM_MODEL)
            return (
                f"I'm not sure what you meant.\nGPT thinks you might be referring to: **{gpt_reply}**\n\nPlease type 'CIC-IDS2018' or 'CESNET'.\n",
                False
            )
            # clarification = clarify_user_intent(user_query, model_name=DEFAULT_LLM_MODEL)
            # return (clarification, False)

    # ------------------- STEP 1: single-step or multistep -------------------
    elif step == 1:
        if "single" in user_lower:
            session_state["analysis_step"] = "single-step"
            session_state["step"] = 2
        elif "multi" in user_lower:
            session_state["analysis_step"] = "multistep"
            session_state["step"] = 2
        else:
            return ("Please type 'single-step' or 'multistep'.\n", False)

        msg = "Do you want 'overall', 'timeslot'"
        if dataset == "CESNET":
            msg += " or 'top10'"
        msg += "?\n(For timeslot, example: 2025-05-04-10:00)\n"
        return (msg, False)

    # ------------------- STEP 2: overall / timeslot / top10 -------------------
    elif step == 2:
        if dataset == "CIC-IDS2018" and "top" in user_lower:
            return ("'top10' is not available for CIC-IDS2018.\n", False)

        if "overall" in user_lower:
            session_state["analysis_type"] = "overall"
            if dataset == "CESNET":
                # Ask "network or ip"
                session_state["step"] = 3
                return (
                    "Do you want to analyze the 'entire network' or a specific 'ip' address?\nType 'network' or 'ip'.\n",
                    False)
            else:
                # Direct result => step=5
                session_state["step"] = 5
                # For CIC, we only have network level data
                session_state["target_scope"] = "network"

                # Make prediction
                result = make_prediction(session_state)
                if result.get("success"):
                    update_prediction_context(state)

                response = format_prediction_response(result, session_state)

                # 若预测成功，则补充 GPT 总结 + 解释
                if result.get("success", False):
                    summary = generate_summary(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    explanation = generate_explanation(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    response += "\n\nSummary:\n" + summary
                    response += "\n\nExplanation:\n" + explanation
                return (response, True)

        elif "timeslot" in user_lower:
            session_state["analysis_type"] = "timeslot"
            if dataset == "CESNET":
                session_state["step"] = 3
                return (
                    "Do you want to analyze the entire 'network' or a specific 'ip' for timeslot?\nType 'network' or 'ip'.\n",
                    False)
            else:
                session_state["target_scope"] = "network"
                session_state["step"] = 4
                return ("Please input a single time point (e.g. 2025-05-04-10:00):\n", False)

        elif "top" in user_lower and dataset == "CESNET":
            session_state["analysis_type"] = "top10"
            session_state["step"] = 5

            # Make prediction
            result = make_prediction(session_state)
            if result.get("success"):
                update_prediction_context(state)

            response = format_prediction_response(result, session_state)

            # 若预测成功，则补充 GPT 总结 + 解释
            if result.get("success", False):
                summary = generate_summary(result, session_state, model_name=DEFAULT_LLM_MODEL)
                explanation = generate_explanation(result, session_state, model_name=DEFAULT_LLM_MODEL)
                response += "\n\nSummary:\n" + summary
                response += "\n\nExplanation:\n" + explanation

            return (response, True)
        else:
            return ("Please choose 'overall', 'timeslot' or 'top10' (only for CESNET).\n", False)

    # ------------------- STEP 3: (CESNET) 'network' or 'ip'? -------------------
    elif step == 3:
        if "network" in user_lower:
            session_state["target_scope"] = "network"
            if session_state["analysis_type"] == "overall":
                # Direct => step=5
                session_state["step"] = 5

                # Make prediction
                result = make_prediction(session_state)
                if result.get("success"):
                    update_prediction_context(state)

                response = format_prediction_response(result, session_state)

                # 若预测成功，则补充 GPT 总结 + 解释
                if result.get("success", False):
                    summary = generate_summary(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    explanation = generate_explanation(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    response += "\n\nSummary:\n" + summary
                    response += "\n\nExplanation:\n" + explanation

                return (response, True)
            elif session_state["analysis_type"] == "timeslot":
                session_state["step"] = 4
                return ("Please input a single time point (e.g. 2025-05-04-10:00):\n", False)
            else:
                return ("Invalid analysis type.\n", False)

        elif "ip" in user_lower:
            session_state["target_scope"] = "ip"
            session_state["step"] = 4

            # Get available IPs from CESNET adapter if available
            available_ips = ["100610", "101", "10125", "10158", "10196", "10197", "10256", "103", "1037"]
            if cesnet_adapter:
                try:
                    available_ips = cesnet_adapter.get_available_ips()
                except:
                    pass

            return (f"Please enter the IP code to analyze (Available IPs: {', '.join(available_ips)}):\n", False)
        else:
            return ("Please type 'network' or 'ip'.\n", False)

    # ------------------- STEP 4: parse IP code or time point -------------------
    elif step == 4:
        atype = session_state["analysis_type"]
        scope = session_state["target_scope"]

        # If scope=ip but user hasn't provided IP code yet:
        if scope == "ip" and not session_state.get("target_ip"):
            # No IP format check, just store user input
            session_state["target_ip"] = user_query
            # If analysis_type=timeslot => ask for time next
            if atype == "timeslot":
                # return ("Got IP code. Now please input a single time point (e.g. 2025-05-04-10:00) (Tips: ):\n", False)
                return (
                f"Got IP code. Now please input a single time point (e.g. 2025-05-04-10:00) [Tips: CESNET time range is 2024-03-13 to 2024-06-03; CICIDS2018 time range is 2018-03-01 03:35 to 2018-03-01 12:59 and 2018-03-02 01:00 to 2018-03-02 12:59]:\n",
                False)
            else:
                # overall => do prediction => step=5
                session_state["step"] = 5

                # Make prediction
                result = make_prediction(session_state)
                if result.get("success"):
                    update_prediction_context(state)

                response = format_prediction_response(result, session_state)

                # 若预测成功，则补充 GPT 总结 + 解释
                if result.get("success", False):
                    summary = generate_summary(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    explanation = generate_explanation(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    response += "\n\nSummary:\n" + summary
                    response += "\n\nExplanation:\n" + explanation

                return (response, True)

        # If timeslot => parse time
        if atype == "timeslot":
            tmatch = parse_single_time_point(user_query)
            if tmatch:
                session_state["time_point"] = tmatch
                session_state["step"] = 5

                # Make prediction
                result = make_prediction(session_state)
                if result.get("success"):
                    update_prediction_context(state)

                response = format_prediction_response(result, session_state)

                # 若预测成功，则补充 GPT 总结 + 解释
                if result.get("success", False):
                    summary = generate_summary(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    explanation = generate_explanation(result, session_state, model_name=DEFAULT_LLM_MODEL)
                    response += "\n\nSummary:\n" + summary
                    response += "\n\nExplanation:\n" + explanation
                return (response, True)
            else:
                return ("Time format invalid. Use e.g. 2025-05-04-10:00\n", False)

        return ("Something unhandled. Type 'restart' or 'exit'.\n", False)

    # ------------------- STEP 5: finished, wait for restart or exit -------------------
    elif step == 5:
        if "restart" in user_lower:
            # Reset
            for k in session_state:
                if k != "step":  # Keep step so we can reset it
                    session_state[k] = None
            session_state["step"] = 0
            return ("Restarting...\nPlease choose dataset: 'CIC-IDS2018' or 'CESNET'\n", False)
        else:
            return ("Type 'restart' or 'exit'.\n", False)

    return ("Unhandled state. Type 'restart' or 'exit'.\n", False)


def handle_client(conn, addr):
    """
    Handle an individual client connection.

    Args:
        conn (socket.socket): Socket connection to the client
        addr (tuple): Client address
    """
    print(f"[+] Connection from {addr}")
    state = get_session_state(addr)

    initial_msg = (
        "Hello, I'm the Oracle Network Traffic Predictor.\n"
        "Which dataset do you want to analyze? 'CIC-IDS2018' or 'CESNET'?\n"
        "----\n"
        "CICIDS2018 is a dataset of network traffic within a controlled experimental environment, recording overall network behavior after aggregation. It does not distinguish between individual IPs or hosts.\n"
        "CESNET-TimeSeries is a summary of traffic on an actual backbone network, with IP/prefix/port granularity, allowing distinction between various traffic sources/destinations.\n"
        "----\n"
        "Type 'exit' at any time to quit.\n"
    )
    conn.sendall(initial_msg.encode('utf-8'))

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print(f"[-] Connection closed by {addr}")
                break

            user_query = data.decode('utf-8').strip()
            print(f"[{addr}] Received: {user_query}")

            if user_query.lower() == "exit":
                conn.sendall(b"Goodbye!\n")
                break

            # Process the user query through the conversation flow
            reply, is_prediction = conversation_flow(user_query, state)
            # conn.sendall(reply.encode('utf-8'))
            # 如果是预测 + 有前导提示，就拆分发送
            if is_prediction and "\nPlease wait...\n" in reply:
                # intro, result_part = reply.split("\nPlease wait...\n", 1)
                # intro_msg = intro + "\nPlease wait...\n"
                # conn.sendall(intro_msg.encode("utf-8"))  # 先发提示
                conn.sendall(reply.encode("utf-8"))  # 先发提示语
                time.sleep(0.3)  # 可选等待
                # conn.sendall(result_part.encode("utf-8"))  # 再发预测结果
                result = make_prediction(state)

                if result.get("success"):
                    update_prediction_context(state)

                final_msg = format_prediction_response(result, state)

                # 如果预测成功，追加 GPT 总结和解释
                if result.get("success", False):
                    summary = generate_summary(result, state, model_name=DEFAULT_LLM_MODEL)
                    explanation = generate_explanation(result, state, model_name=DEFAULT_LLM_MODEL)
                    final_msg += "\n\nSummary:\n" + summary
                    final_msg += "\n\nExplanation:\n" + explanation
                print("[DEBUG] 发送预测结果：", final_msg)
                conn.sendall(final_msg.encode("utf-8"))  # 发预测结果（关键！！）
            else:
                conn.sendall(reply.encode('utf-8'))

            # Log prediction results
            if is_prediction:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dataset = state["dataset"] or "Unknown"
                analysis_type = state["analysis_type"] or "Unknown"
                print(f"[{timestamp}] Prediction made for {addr}: {dataset} - {analysis_type}")

        except Exception as e:
            print(f"[!] Error with {addr}: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"An error occurred: {str(e)}\nPlease try again or type 'restart'.\n"
            try:
                conn.sendall(error_msg.encode('utf-8'))
            except:
                pass  # Connection may be broken
            break

    reset_session_state(addr)
    conn.close()
    print(f"[*] Disconnected from {addr}")


def run_server(host='127.0.0.1', port=50011):
    """
    Run the Oracle server.

    Args:
        host (str): Host to bind to
        port (int): Port to bind to
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(5)
        print(f"[*] Oracle server started on {host}:{port}, waiting for connections...")

        # Log adapter status
        print(f"[*] CESNET adapter: {'Active' if cesnet_adapter else 'Not available'}")
        print(f"[*] CIC-IDS2018 adapter: {'Active' if cic_adapter else 'Not available'}")

        while True:
            try:
                conn, addr = s.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr))
                thread.daemon = True  # Make thread a daemon so it exits when main thread exits
                thread.start()
            except KeyboardInterrupt:
                print("\n[*] Shutting down server...")
                break
            except Exception as e:
                print(f"[!] Error accepting connection: {e}")


if __name__ == "__main__":
    run_server()