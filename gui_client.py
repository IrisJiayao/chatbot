import tkinter as tk
import socket
import threading

class ChatClientUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Oracle Chatbot - Connecting...")
        self.master.geometry("500x400")

        # ====== 上方的消息显示区域 ======
        self.text_area = tk.Text(self.master, wrap='word', state='disabled')
        self.text_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # ====== 颜色标签 ======
        self.text_area.tag_config("blue", foreground="blue")
        self.text_area.tag_config("green", foreground="green")
        self.text_area.tag_config("red", foreground="red")

        # ====== 输入区域 ======
        self.entry_frame = tk.Frame(self.master)
        self.entry_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.user_input = tk.Entry(self.entry_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        # ====== Reconnect 按钮（开始隐藏）======
        self.reconnect_button = tk.Button(self.master, text="Reconnect", command=self.reconnect)
        self.reconnect_button.pack(pady=5)
        self.reconnect_button.pack_forget()  # 初始隐藏

        # ====== Socket相关 ======
        self.host = '127.0.0.1'
        self.port = 50009

        self.client_socket = None
        self.running = True

        # 尝试连接服务器
        self.connect_to_server()

    def connect_to_server(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))

            # ✅ 成功连接
            self.append_text("Connected to Oracle server.\n", tag="blue")

            self.master.title("Oracle Chatbot")

            # 启动后台接收线程
            receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
            receive_thread.start()

            # 启用输入和按钮
            self.user_input.config(state='normal')
            self.send_button.config(state='normal')
            self.reconnect_button.pack_forget()

        except Exception as e:
            # ❌ 连接失败
            self.append_text("❌ Unable to connect to Oracle server. Please check if the server is running.\n", tag="red")
            self.append_text(f"Error: {e}\n", tag="red")
            self.master.title("Oracle Chatbot - ❌ Offline")

            # 禁用输入
            self.user_input.config(state='disabled')
            self.send_button.config(state='disabled')
            self.reconnect_button.pack()  # 显示重连按钮

    def receive_loop(self):
        while self.running:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    self.append_text("[CLIENT] Server closed connection.\n", tag="red")
                    self.running = False
                    break
                message = data.decode('utf-8')
                self.append_text(f"Oracle: {message}\n")
            except OSError:
                break
            except Exception as e:
                self.append_text(f"Receive error: {e}\n", tag="red")
                break

    def send_message(self, event=None):
        text = self.user_input.get().strip()
        if text:
            try:
                self.client_socket.sendall(text.encode('utf-8'))
                self.append_text(f"You: {text}\n", tag="green")
                self.user_input.delete(0, tk.END)
            except Exception as e:
                self.append_text(f"Send failed: {e}\n", tag="red")

    def append_text(self, msg, tag=None):
        self.text_area.config(state='normal')
        if tag:
            self.text_area.insert(tk.END, msg, tag)
        else:
            self.text_area.insert(tk.END, msg)
        self.text_area.config(state='disabled')
        self.text_area.see(tk.END)

    def reconnect(self):
        self.append_text("\nReconnecting...\n", tag="blue")
        self.connect_to_server()

    def on_closing(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        self.master.destroy()

def main():
    root = tk.Tk()
    app = ChatClientUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

