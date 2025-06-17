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
#
# 这里我们去除了原先的 IP 正则判断，改为直接存储用户输入。
# 例如用户输入 "100610" 或 "1037" 都直接接受为 IP 代码。

import socket
import threading
import random
import nltk
import string
import re

# If not downloaded:
# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))
puncts = set(string.punctuation)

def fake_traffic_predict():
    """Placeholder function returning a random traffic value."""
    return round(random.uniform(100, 500), 2)

def real_tft_predict():
    """Placeholder for actual model inference."""
    return fake_traffic_predict()

session_states = {}

def get_session_state(addr):
    """Retrieve or create a session state dictionary for a given client address."""
    if addr not in session_states:
        session_states[addr] = {
            "step": 0,
            "dataset": None,         # "CIC-IDS2018" or "CESNET"
            "analysis_step": None,   # "single-step" or "multistep"
            "analysis_type": None,   # "overall", "timeslot", "top10"
            "time_point": None,      # string like "2025-05-04-10:00"
            "target_scope": None,    # "network" or "ip"
            "target_ip": None        # user input for IP code, e.g. "100610"
        }
    return session_states[addr]

def reset_session_state(addr):
    """Remove the session state for a client."""
    session_states.pop(addr, None)

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

def conversation_flow(user_query: str, session_state: dict):
    """
    Main state machine for multi-step dialogue.
    """

    step = session_state["step"]
    dataset = session_state["dataset"]
    analysis_step = session_state["analysis_step"]
    analysis_type = session_state["analysis_type"]
    time_point = session_state["time_point"]
    target_scope = session_state["target_scope"]
    target_ip = session_state["target_ip"]

    user_lower = user_query.lower().strip()

    # ------------------- STEP 0: Choose dataset -------------------
    if step == 0:

        if "cic" in user_lower:
            session_state["dataset"] = "CIC-IDS2018"
            session_state["step"] = 1
            return ("You chose CIC-IDS2018. Do you want 'single-step' or 'multistep'? (Multistep can predict network traffic over a larger time horizon but take longer to analyse; single-step predict a smaller time horizon but take less time to analyse.)\n", False)
        elif "cesnet" in user_lower:
            session_state["dataset"] = "CESNET"
            session_state["step"] = 1
            return ("You chose CESNET. Do you want 'single-step' or 'multistep'? (Multistep can predict network traffic over a larger time horizon but take longer to analyse; single-step predict a smaller time horizon but take less time to analyse.)\n", False)
        else:
            return ("Please choose dataset: 'CIC-IDS2018' or 'CESNET'.\n", False)

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
            return ("'top10' is not available for CIC.\n", False)

        if "overall" in user_lower:
            session_state["analysis_type"] = "overall"
            if dataset == "CESNET":
                # ask "network or ip"
                session_state["step"] = 3
                return ("Do you want to analyze the 'entire network' or a specific 'ip' address?\nType 'network' or 'ip'.\n", False)
            else:
                # direct result => step=5
                session_state["step"] = 5
                pred = real_tft_predict()
                msg = f"[CIC Overall] Predicted traffic: {pred} Mbps (fake)\nType 'restart' or 'exit'.\n"
                return (msg, False)

        elif "timeslot" in user_lower:
            session_state["analysis_type"] = "timeslot"
            if dataset == "CESNET":
                session_state["step"] = 3
                return ("Do you want to analyze the entire 'network' or a specific 'ip' for timeslot?\nType 'network' or 'ip'.\n", False)
            else:
                session_state["target_scope"] = "network"
                session_state["step"] = 4
                return ("Please input a single time point (e.g. 2025-05-04-10:00):\n", False)

        elif "top" in user_lower and dataset == "CESNET":
            session_state["analysis_type"] = "top10"
            session_state["step"] = 5
            pred = real_tft_predict()
            msg = f"[CESNET - Top10 IPs] Predicted traffic: {pred} Mbps (fake)\nType 'restart' or 'exit'.\n"
            return (msg, False)
        else:
            return ("Please choose 'overall', 'timeslot' or 'top10'(only for CESNET if not CIC).\n", False)

    # ------------------- STEP 3: (CESNET) 'network' or 'ip'? -------------------
    elif step == 3:
        if "network" in user_lower:
            session_state["target_scope"] = "network"
            if session_state["analysis_type"] == "overall":
                # direct => step=5
                session_state["step"] = 5
                pred = real_tft_predict()
                msg = f"[CESNET - overall network] Predicted traffic: {pred} Mbps (fake)\nType 'restart' or 'exit'.\n"
                return (msg, False)
            elif session_state["analysis_type"] == "timeslot":
                session_state["step"] = 4
                return ("Please input a single time point (e.g. 2025-05-04-10:00):\n", False)
            else:
                return ("Invalid analysis type.\n", False)

        elif "ip" in user_lower:
            session_state["target_scope"] = "ip"
            session_state["step"] = 4
            return ("Please enter the IP code to analyze (You can predict network traffic for the following IPs: 100610, 101, 10125, 10158, 10196, 10197, 10256, 103, 1037):\n", False)
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
                return ("Got IP code. Now please input a single time point (e.g. 2025-05-04-10:00):\n", False)
            else:
                # overall => do fake predict => step=5
                session_state["step"] = 5
                pred = real_tft_predict()
                return (f"[CESNET overall + IP {user_query}] Predicted: {pred} Mbps (fake)\nType 'restart' or 'exit'.\n", False)

        # If timeslot => parse time
        if atype == "timeslot":
            tmatch = parse_single_time_point(user_query)
            if tmatch:
                session_state["time_point"] = tmatch
                session_state["step"] = 5
                pred = real_tft_predict()
                ipinfo = ""
                if scope == "ip" and session_state.get("target_ip"):
                    ipinfo = f", IP={session_state['target_ip']}"
                return (f"[Timeslot {tmatch}{ipinfo}] Predicted traffic: {pred} Mbps\nType 'restart' or 'exit'.\n", False)
            else:
                return ("Time format invalid. Use e.g. 2025-05-04-10:00\n", False)

        return ("Something unhandled. Type 'restart' or 'exit'.\n", False)

    # ------------------- STEP 5: finished, wait for restart or exit -------------------
    elif step == 5:
        if "restart" in user_lower:
            # reset
            for k in session_state:
                session_state[k] = None
            session_state["step"] = 0
            return ("Restarting...\nPlease choose dataset: 'CIC-IDS2018' or 'CESNET'\n", False)
        else:
            return ("Type 'restart' or 'exit'.\n", False)

    return ("Unhandled state. Type 'restart' or 'exit'.\n", False)

def handle_client(conn, addr):
    print(f"[+] Connection from {addr}")
    state = get_session_state(addr)

    initial_msg = (
        "Hello, I'm the Oracle.\n"
        "Which dataset do you want to analyze? 'CIC-IDS2018' or 'CESNET'?\n"
        "----CICIDS2018 is a single area of network traffic within the experimental environment, recording only the overall network behaviour after aggregation, unable to distinguish between the behaviour of individual IPs or hosts within the network; CESNET-TimeSeries is an aggregation of traffic on the actual backbone network, usually with IP / prefix / port as granularity, which can distinguish between various types of source / destination traffic. CESNET-TimeSeries is a summary of traffic on the actual backbone network.\n"
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

            reply, _ = conversation_flow(user_query, state)
            conn.sendall(reply.encode('utf-8'))

        except Exception as e:
            print(f"[!] Error with {addr}: {e}")
            break

    reset_session_state(addr)
    conn.close()
    print(f"[*] Disconnected from {addr}")

def run_server():
    HOST = '127.0.0.1'
    PORT = 50009

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"[*] Server started on {HOST}:{PORT}, waiting for connections...")

        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()

if __name__ == "__main__":
    run_server()
