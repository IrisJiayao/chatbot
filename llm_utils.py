import openai
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

client = openai.OpenAI(api_key="")
# ==============================
# 🔹 GPT（OpenAI）调用函数
# ==============================
def call_openai(prompt, model="gpt-3.5-turbo", temperature=0.2, max_tokens=1024):
    try:

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI Error] {e}"

# ==============================
# 🔹 DeepSeek 调用函数（本地推理）
# ==============================
def call_deepseek(prompt, model_name="deepseek-ai/deepseek-llm-1.3b-chat"):
    try:
        # 推荐在首次调用外部预载
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

        output = pipe(prompt, max_new_tokens=512, do_sample=True)[0]["generated_text"]
        return output.strip()
    except Exception as e:
        return f"[DeepSeek Error] {e}"

# ==============================
# 🔹 统一入口：call_llm
# ==============================
def call_llm(prompt, model="gpt-3.5", **kwargs):
    print(f"\n[LLM] Calling model: {model}")

    if model.startswith("gpt"):
        return call_openai(prompt, model=model, **kwargs)

    elif model == "deepseek":
        return call_deepseek(prompt)

    else:
        return f"[LLM Error] Unsupported model: {model}"

def load_prompt_template(name: str, input_dict: dict) -> str:
    """
    加载 prompts/name.txt 并将 {{KEY}} 替换成 input_dict["KEY"]
    """
    try:
        path = f"prompt/{name}.txt"
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
        for key, value in input_dict.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        return template
    except Exception as e:
        return f"[Prompt 加载失败: {e}]"

def call_llm_conversational(user_input: str, session_state: dict, model: str = "gpt") -> str:
    """
    调用 GPT/LLM 并保留历史多轮对话。
    """
    try:
        # 初始化对话历史
        if "history" not in session_state:
            session_state["history"] = []

        # 添加用户输入
        session_state["history"].append({"role": "user", "content": user_input})

        # 调用 GPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or other
            messages=session_state["history"],
            temperature=0.4,
            max_tokens=400
        )

        # 提取 assistant 回复并加入历史
        reply = response.choices[0].message.content.strip()
        session_state["history"].append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        return f"⚠️ Error: {e}"

def finalize_prediction_parameters(extracted: dict, context: dict) -> dict:
    """
    用历史上下文补全提取参数中缺失的部分。

    Args:
        extracted (dict): 来自 extract_parameters_from_nl 的 LLM 提取结果（可能有空字段）
        context (dict): 上一次预测的完整上下文（last_prediction_context）

    Returns:
        dict: 最终预测参数（dataset, analysis_type, target_ip, time_point）
    """
    return {
        "dataset": extracted.get("dataset") or context.get("dataset"),
        "analysis_type": extracted.get("analysis_type") or context.get("analysis_type"),
        "target_ip": extracted.get("target_ip") or context.get("target_ip"),
        "time_point": extracted.get("time_point") or context.get("time_point"),
    }


def generate_caption_from_chart(dataset, ip, time_point, model="gpt"):
    prompt = load_prompt_template("chart_caption", {
        "dataset": dataset,
        "ip": ip,
        "time": time_point
    })
    return call_llm(prompt, model=model)

def summarize_multiple_predictions(result_list, model="gpt"):
    formatted_records = ""
    for r in result_list:
        formatted_records += f"- IP: {r['target_ip']}, Time: {r['time_point']}, Predicted: {r['prediction']}, Actual: {r['actual']}\n"

    prompt = load_prompt_template("summarize_multiple_predictions", {
        "records": formatted_records
    })
    return call_llm(prompt, model=model)
