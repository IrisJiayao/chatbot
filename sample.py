import pandas as pd

# 读取 tshark 解析后的 CSV
input_csv = "D:/PythonProject/chatbot/dataset/mawi_pcap_data/0303extracted_tcp_udp.csv"
output_csv = "D:/PythonProject/chatbot/dataset/mawi_pcap_data/0301sampled_traffic.csv"

df = pd.read_csv(input_csv)

# 处理时间戳（转换为标准时间格式）
df["frame.time_epoch"] = pd.to_datetime(df["frame.time_epoch"], unit="s")

# 进行 10s 采样
df_resampled = df.resample("10S", on="frame.time_epoch").agg({
    "ip.src": "nunique",   # 10s 内唯一源 IP 数
    "ip.dst": "nunique",   # 10s 内唯一目标 IP 数
    "frame.len": "sum",    # 10s 内总流量（字节）
    "frame.number": "count",  # 10s 内的包数量
    "ip.proto": "max",     # 10s 内最常见协议
    "tcp.srcport": "nunique",  # 10s 内的唯一源 TCP 端口数
    "tcp.dstport": "nunique",  # 10s 内的唯一目标 TCP 端口数
    "tcp.flags": "max",        # 10s 内出现的最大 TCP 标志值
    "tcp.window_size": "mean"  # 10s 内 TCP 窗口大小的平均值
})

# 重命名列
df_resampled.rename(columns={
    "ip.src": "unique_src_ips",
    "ip.dst": "unique_dst_ips",
    "frame.len": "total_bytes",
    "frame.number": "packet_count",
    "ip.proto": "common_protocol",
    "tcp.srcport": "unique_src_ports",
    "tcp.dstport": "unique_dst_ports",
    "tcp.flags": "tcp_flag_max",
    "tcp.window_size": "tcp_window_avg",
}, inplace=True)

# 保存到 CSV
df_resampled.to_csv(output_csv, index=False)
print(f"\n✅ 10s 采样完成，结果已保存到 {output_csv}")
