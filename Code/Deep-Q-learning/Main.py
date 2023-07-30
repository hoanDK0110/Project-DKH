import networkx as nx
import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

PHY = nx.read_gml("./data_import/data_PHY/atlanta.gml")
SFC = nx.read_gml("./data_import/data_SFC/SFC_graph.gml")


class SFCMappingEnvironment:
    def __init__(self):
        # Thiết lập mạng vật lý
        self.PHY_nodes = list(PHY.nodes())
        self.PHY_weights_node = [PHY.nodes[node]['weight'] for node in self.PHY_nodes]
        self.PHY_array = nx.adjacency_matrix(PHY).toarray()

        # Thiết lập mạng SFC
        self.SFC_nodes = list(SFC.nodes())
        self.SFC_weights_node = [SFC.nodes[node]['weight'] for node in self.SFC_nodes]
        self.SFC_array = nx.adjacency_matrix(SFC).toarray()

        # Thiết lập không gian trạng thái và hành động
        self.state_space = list(range(len(self.SFC_nodes)))
        self.action_space = list(range(len(self.PHY_nodes)))
         # Phương thức cập nhật trọng số của các node PHY sau khi mapping
    def update_PHY_node_weights(self, mapping_pairs):
        for node_SFC, node_PHY in mapping_pairs:
            self.PHY_weights_node[node_PHY] -= self.SFC_weights_node[node_SFC]

def find_max_values(trained_model, state_space, action_space):
    mapping_pairs = []
    selected_actions = set()
    total_q_value = 0

    for sfc_node in state_space:
        q_values = trained_model.predict(np.eye(len(state_space))[sfc_node].reshape(1, len(state_space)), verbose=0)

        available_actions = [action for action in range(len(action_space)) if action not in selected_actions]

        if len(available_actions) == 0:
            print("Cannot map further.")
            break

        action = available_actions[np.argmax(q_values[0, available_actions])]
        max_q_value = np.max(q_values[0, available_actions])

        mapping_pairs.append([sfc_node, action])
        selected_actions.add(action)
        total_q_value += max_q_value

    return mapping_pairs, total_q_value
def dijkstra(graph, start, end, weight_requirement):
    num_nodes = len(graph)
    distances = np.full(num_nodes, np.inf)  # Khởi tạo khoảng cách ban đầu là vô cùng
    distances[start] = 0  # Khoảng cách từ nút bắt đầu đến chính nó là 0

    visited = set()  # Tập các nút đã được duyệt
    previous = np.full(num_nodes, None)  # Mảng lưu các nút trước đó trên đường đi ngắn nhất
    updated_graph = [list(row) for row in graph]
    # Duyệt qua tất cả các nút
    for _ in range(num_nodes):
        # Tìm nút có khoảng cách nhỏ nhất và chưa được duyệt
        min_distance = np.inf
        min_node = None
        for node in range(num_nodes):
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        if min_node is None:
            break  # Không có đường đi từ nút bắt đầu đến nút kết thúc

        visited.add(min_node)  # Đánh dấu nút đã được duyệt

        # Kiểm tra nếu đã đến nút kết thúc
        if min_node == end:
            path = []
            node = end
            while node is not None:
                path.insert(0, node)
                node = previous[node]
                
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                updated_graph[node1][node2] -= weight_requirement

            total_weight = 0
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                total_weight += graph[node1][node2]
                
            return len(path), updated_graph, total_weight#, updated_graph  # Trả về số lượng nút phải đi qua, đường đi và tổng trọng số cạnh

        # Cập nhật khoảng cách và nút trước đó cho các nút kề
        for neighbor in range(num_nodes):
            if neighbor not in visited and graph[min_node][neighbor] >= weight_requirement:
                new_distance = distances[min_node] + graph[min_node][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = min_node

    return -1, updated_graph, 0#, graph  # Không tìm thấy đường đi từ nút bắt đầu đến nút kết thúc


class DQNAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.num_episodes = 20

        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
 

        self.update_target_nn = 10

        # Tạo mạng main network và target network
        self.main_network = self.build_neural_network(len(state_space), len(action_space))
        self.target_network = self.build_neural_network(len(state_space), len(action_space))
        self.update_target_network()

        # Bộ nhớ tái trải nghiệm (replay buffer)
        self.replay_buffer = deque(maxlen=5000)

    # Update mạng target = main
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def remember_exp(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn_from_replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in batch:
            '''print("state in batch:", state)
            print("action in batch:", action)
            print("reward in batch:", reward)
            print("next_state in batch:", next_state)
            print("done in batch:", done)'''
            #state = tf.convert_to_tensor(state, dtype=tf.float32)
            #next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
            target = reward
            if not done:
                #print(self.target_network.predict(next_state))
                q_values = self.target_network.predict(next_state, verbose=0)
                q_future = np.amax(q_values)
                #print("q_future: ", q_future)
                target = reward + self.discount_factor * q_future

            target_q_values = self.main_network.predict(state, verbose=0)
            #print("target_q_values:", target_q_values)
            target_q_values[0][action] = target
            self.main_network.fit(state, target_q_values, epochs=1, verbose=0)
        
    # Xây dựng mạng neural
    def build_neural_network(self, input_dim, output_dim):
        model = Sequential()
        model.add(Dense(32, input_dim=input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    def choice_action(self, current_state_array, current_action_space):
        # Khám phá
        if np.random.rand() < agent.epsilon:
            # Chọn hành động bất kì trong không gian hành động 
            action = np.random.choice(current_action_space) 
            current_action_space.remove(action) 
        # Khai thác
        else:
            q_values = self.main_network.predict(current_state_array, verbose=0)
            action = np.argmax(q_values[0])
            #print("Action exploitation: ", action)
        return action, current_action_space
    
    def deep_q_learning(self,env, batch_size=32):
        reward_array = np.array([])
        q_array = np.array([])
        # Quá trình training
        for episode in range(self.num_episodes):
            current_state_space = env.state_space.copy()# Copy không gian trạng thái
            current_action_space = env.action_space.copy() # Copy không gian hành động
            current_PHY_array = env.PHY_array.copy()# Copy ma trận kề của PHY

            selected_action_space = np.array([]) # Lưu trữ các hành động đã chọn
            #selected_action_space_check = np.array([]) # Lưu trữ các hành động đã thử
            selected_state_array = np.array([]) # Lưu trữ các state đã chọn

            total_reward = 0 # Tính reward nhận được ở mỗi episode
            # Cập nhật mạng target network sau một số episodes
            if episode % self.update_target_nn == 0 and episode != 0:
                self.update_target_network()
            
            done = False
            while not done:
                if len(current_state_space) == 0: # Check xem đã hết trạng thái chưa
                    break 
                # Chọn trạng thái
                current_state = current_state_space.pop(0) # Lấy trạng thái từ trong mảng
                current_state_array = np.eye(len(env.state_space))[current_state]
                current_state_array = current_state_array.reshape(1, len(env.state_space))
                #print("current_state: ",current_state)
                selected_state_array = np.append(selected_state_array, current_state)# lưu trạng thái vừa chọn vào mảng để tính toán reward
                #print("selected_state_array: ",selected_state_array)
        
                # Chọn hành động (chọn node PHY) thỏa mãn cap of PHY node > cap of SFC node và link map giữa 2 node PHY > link giữa 2 SFC liên tiếp
                while 1:
                    
                    action, current_action_space = agent.choice_action(current_state_array, current_action_space)
                    # Kiểm tra hành động được chọn ở trên (action)
                    if action not in selected_action_space and env.PHY_weights_node[action] >= env.SFC_weights_node[current_state]: # Kiểm tra hành động đó có được chọn trước đo hay chưa và cap của SFc >= cap của PHY
                        if len(selected_state_array) == 1: # Nếu là hành động động đầu tiên được chọn thì được chọn và lưu trữ
                            selected_action_space = np.append(selected_action_space,action)
                            break
                        else: # Nếu là hành động thứ 3 trở đi được chọn
                            
                            num_hop, current_PHY_array, total_weight_edge = dijkstra(current_PHY_array, int(selected_action_space[-1])  , action, env.SFC_array[current_state - 1][current_state]) # Tính toán xem có đường đi có thỏa mãn trọng số và cập nhật ma trận kề của PHY khi đã tìm thấy đường đi ngắn nhất
                            
                            if num_hop != - 1: # Nếu có đường đi thì hành động sẽ được chọn và lưu trữ
                                selected_action_space = np.append(selected_action_space,action)
                                break
                    
                    if len(current_action_space) == 0:
                        print("can't map ")
                        return self.main_network, current_PHY_array, 0
                        
                if len(selected_state_array) == 1: # Tính toán Reward cho hành động đầu tiên
                    reward = 10000 - (env.PHY_weights_node[action] - env.SFC_weights_node[current_state])
                elif 1 < len(selected_state_array) <  len(env.state_space) :  # Không phải trạng thái đầu và trạng thái cuối
                    reward = 10000 - (env.PHY_weights_node[action] - env.SFC_weights_node[current_state]) - 5 * num_hop - 3 * (total_weight_edge - (num_hop - 1) * env.SFC_array[current_state - 1][current_state])
                    
                else: # Tính toán Reward cho hành động từ lần thứ 2
                    done = True
                    reward = 10000 - (env.PHY_weights_node[action] - env.SFC_weights_node[current_state]) - 5 * num_hop - 3 * (total_weight_edge - (num_hop - 1) * env.SFC_array[current_state - 1][current_state])
                
                
                next_state = current_state + 1 if current_state_space else current_state
                next_state_array = np.eye(len(env.state_space))[next_state]
                next_state_array = next_state_array.reshape(1, len(env.state_space))      
                self.remember_exp(current_state_array, action, reward, next_state_array, done)
                total_reward += reward 
                
            # Học từ bộ nhớ tái trải nghiệm   
            self.learn_from_replay(batch_size)
            #Lưu trữ giá trị reward nhận được ở mỗi episode
            reward_array = np.append(reward_array,total_reward) 
            #print(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward}")
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon = agent.epsilon * agent.epsilon_decay
            _, total_q_value = find_max_values(self.main_network, env.state_space, env.action_space)
            q_array = np.append(q_array, total_q_value)
            
        #np.savetxt('./Deep-Q-Learning/data_reward.txt', reward_array)  # Xuất giá trị ra file
        np.savetxt('./Deep-Q-Learning/data/data_reward.txt', reward_array) # Xuất giá trị ra file
        np.savetxt('./Deep-Q-Learning/data/data_q.txt', q_array) # Xuất giá trị ra file
        
        return self.main_network, current_PHY_array, 1 

# Chương trình bắt đầu từ đây
env = SFCMappingEnvironment()
agent = DQNAgent(env.state_space, env.action_space)
agent.deep_q_learning(env)

check = 1
while check:
    # Gọi vào thuật toán
    trained_model, current_PHY_array, check = agent.deep_q_learning(env)

    if check != 0:
        mapping_pairs, total_q_value = find_max_values(trained_model, env.state_space, env.action_space)
        print("Mapping: ", mapping_pairs)
        for i in range(len(mapping_pairs) - 1):
            start, end = mapping_pairs[i][1], mapping_pairs[i + 1][1]
            _, env.PHY_array, _ = dijkstra(env.PHY_array, start, end, env.SFC_array[i][0])

        env.update_PHY_node_weights(mapping_pairs)