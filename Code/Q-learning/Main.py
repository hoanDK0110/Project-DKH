import numpy as np
import networkx as nx

num_episodes = 4000
learning_rate = 0.1
discount_factor = 1
exploration_rate = 0.9
c_start_ep_epsilon_decay = 1
c_end_ep_epsilon_decay = num_episodes
v_epsilon_decay = exploration_rate / (c_end_ep_epsilon_decay - c_start_ep_epsilon_decay)

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

    # Phương thức cập nhật trọng số cạnh của các cạnh liên kết giữa các node PHY sau khi mapping
    def update_PHY_edge_weights(self, mapping_pairs):
        for node_SFC, node_PHY in mapping_pairs:
            # Duyệt qua các node PHY kề với node_PHY được mapping
            for neighbor_PHY in self.PHY_nodes:
                if self.PHY_array[node_PHY][neighbor_PHY] > 0:
                    # Trừ đi trọng số cạnh liên kết giữa các node PHY tương ứng với trọng số nút SFC đã mapping
                    self.PHY_array[node_PHY][neighbor_PHY] -= self.SFC_array[node_SFC][node_SFC + 1]

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

def find_max_values(array):
    # Đếm số hàng và số cột của mảng
    num_rows = len(array)
    num_cols = len(array[0])
    
    max_values = []  # Danh sách để lưu trữ giá trị lớn nhất của mỗi hàng
    max_positions = []  # Danh sách để lưu trữ vị trí của giá trị lớn nhất của mỗi hàng

    chosen_columns = set()  # Tập hợp các cột đã được chọn
    
    for row_idx in range(num_rows):
        max_value = None  # Giá trị lớn nhất của hàng hiện tại
        max_position = None  # Vị trí của giá trị lớn nhất trong hàng hiện tại
        
        for col_idx in range(num_cols):
            if col_idx not in chosen_columns:
                value = array[row_idx][col_idx]  # Giá trị của phần tử hiện tại
                
                if max_value is None or value > max_value:
                    max_value = value
                    max_position = (row_idx, col_idx)

        if max_value is not None:
            chosen_columns.add(max_position[1])  # Thêm cột đã chọn vào tập hợp
            
            max_values.append(max_value)  # Thêm giá trị lớn nhất vào danh sách
            max_positions.append(max_position)  # Thêm vị trí vào danh sách

    return max_values, max_positions

def sum_q(arr):
    # Tạo một mảng tạm để lưu trữ cột đã được chọn
    selected_cols = []

    # Tính tổng giá trị lớn nhất của mỗi hàng với điều kiện giá trị lớn nhất của cột đã được chọn
    sum_max_values = 0
    for row in arr:
        max_value = np.max(row[np.logical_not(np.isin(row, selected_cols))])
        sum_max_values += max_value
        selected_cols.append(max_value)

    return sum_max_values

def calculator_q(current_state_space, q_table, selected_action_space, current_state, action, reward):
    if current_state_space: # Tính Q_value cho trạng thái không phải cuối cùng
        # Copy mảng trạng thái kế
        array_next_state = q_table[current_state_space[0]].copy()
        # Tìm giá trị max của trạng thái tiếp theo mà không có các hàng động đã chọn
        max_next_state = np.max(array_next_state[np.logical_not(np.isin(np.arange(len(array_next_state)), selected_action_space))])
        # Tính Q_value
        q_table[current_state][action] = (1 - learning_rate) * q_table[current_state][action] + learning_rate * (reward + discount_factor * max_next_state)
    else: # Tính Q_value cho trạng thái cuối cùng
        q_table[current_state][action] = (1 - learning_rate) * q_table[current_state][action] + learning_rate * reward
        

# Thuật toán Q-Learning
def q_learning(env, num_episodes, learning_rate, discount_factor, exploration_rate):
    # Khởi tạo bảng Q table là ma trận 3(số node SFC) x 6(số node PHY) có tất cả giá trị bằng 0
    q_table = np.zeros((len(env.state_space), len(env.action_space)))
    reward_array = np.array([])
    q_array = np.array([])
    # Quá trình training
    for episode in range(num_episodes):
        current_state_space = env.state_space.copy()# Copy không gian trạng thái
        current_action_space = env.action_space.copy() # Copy không gian hành động
        current_PHY_array = env.PHY_array.copy()# Copy ma trận kề của PHY

        selected_action_space = np.array([]) # Lưu trữ các hành động đã chọn
        selected_action_space_check = np.array([]) # Lưu trữ các hành động đã thử
        selected_state_array = np.array([]) # Lưu trữ các state đã chọn

        sum_reward = 0 # Tính reward nhận được ở mỗi episode
        while 1:
            if len(current_state_space) == 0: # Check xem đã hết trạng thái chưa
                break 
            # Chọn trạng thái
            current_state = current_state_space.pop(0) # Lấy trạng thái từ trong mảng
            #print("current_state: ",current_state)
            selected_state_array = np.append(selected_state_array, current_state)# lưu trạng thái vừa chọn vào mảng để tính toán reward
            #print("selected_state_array: ",selected_state_array)
    
            # Chọn hành động (chọn node PHY) thỏa mãn cap of PHY node > cap of SFC node và link map giữa 2 node PHY > link giữa 2 SFC liên tiếp
            while 1:
                # Khám phá
                if np.random.rand() < exploration_rate:
                    action = np.random.choice(current_action_space) # Chọn hành động bất kì trong không gian hành động 
                    current_action_space.remove(action) 
                # Khai thác
                else:
                    action = np.argmax(q_table[current_state]) # Chọn hành động có giá trị Q lớn nhất trong bảng Q table
                
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
                    return q_table, current_PHY_array, 0
                    
                        
           
            if len(selected_state_array) == 1: # Tính toán Reward cho hành động đầu tiên
                reward = 1000 - (env.PHY_weights_node[action] - env.SFC_weights_node[current_state])
                calculator_q(current_state_space, q_table, selected_action_space, current_state, action, reward)
            else: # Tính toán Reward cho hành động từ lần thứ 2
                reward = 1000 - (env.PHY_weights_node[action] - env.SFC_weights_node[current_state]) - 5 * num_hop
                calculator_q(current_state_space, q_table, selected_action_space, current_state, action, reward)
            sum_reward += reward   
            
        
        reward_array = np.append(reward_array,sum_reward) #Lưu trữ giá trị reward nhận được ở mỗi episode
        total_q_value = sum_q(q_table)
        q_array = np.append(q_array, total_q_value)
        if c_end_ep_epsilon_decay >= episode > c_start_ep_epsilon_decay:
            exploration_rate = exploration_rate - v_epsilon_decay
        
    np.savetxt('./Q_Learning/data/data_q.txt', q_array) # Xuất giá trị ra file
    np.savetxt('./Q_Learning/data/data_reward.txt', reward_array) # Xuất giá trị ra file
    
    
    
    return q_table, current_PHY_array, 1 # Trả về bảng Q_table



#================================================================================================================#
#================================================================================================================#
#================================================================================================================#
#================================================================================================================#


# Chương trình bắt đầu từ đây
env = SFCMappingEnvironment()
check = 1
while check:
    # Gọi vào thuật toántoán
    q_table, current_PHY_array, check = q_learning(env, num_episodes, learning_rate, discount_factor, exploration_rate)
 
    #print("Q table co gia tri la:" , q_table)
    if check != 0:
        max_values, mapping_pairs = find_max_values(q_table)
        #print("max_values: ", max_values)
        print("max_positions: ", mapping_pairs)  
        for i in range(len(mapping_pairs) - 1):
            start, end = mapping_pairs[i][1], mapping_pairs[i+1][1]  
            #print(end, start)  # Kết quả: [12, 5]
            env.SFC_array[i][0]
            #print(env.SFC_array[i][i+1])  # Kết quả: [12, 5]
            
            _, env.PHY_array, _ = dijkstra(env.PHY_array, start , end, env.SFC_array[i][0])
                                                        
        env.update_PHY_node_weights(mapping_pairs)
    
    #env.PHY_array = current_PHY_array 