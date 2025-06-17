import socket

def run_client():
    HOST = '127.0.0.1'
    PORT = 50011

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to Oracle server. Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            s.sendall(user_input.encode('utf-8'))
            data = s.recv(1024)
            if not data:
                print("[CLIENT] Server closed connection.")
                break

            print("Oracle:", data.decode('utf-8'))

if __name__ == "__main__":
    run_client()
