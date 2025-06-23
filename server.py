import socket
import struct
import numpy as np

def start_server(ip="127.0.0.1", port=5050):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(1)
    print(f"Waiting for UE5 to connect at {ip}:{port}...")
    conn, addr = server.accept()
    print(f"Connected by {addr}")
    return conn

def recv_state(conn, input_dim):
    data = conn.recv(4 * input_dim)
    state = struct.unpack(f'{input_dim}f', data)
    return np.array(state, dtype=np.float32)

def send_action(conn, action):
    data = struct.pack(f'{len(action)}f', *action)
    conn.sendall(data)

def recv_reward_done(conn):
    data = conn.recv(8)  # 1 float reward + 1 bool done
    reward, done = struct.unpack('f?', data)
    return reward, done