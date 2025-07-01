import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import concurrent.futures

# 全局变量，用于进程间共享数据
_shared_data = None

def read_excel_data(file_path):
    """读取Excel文件并转换为整数数组"""
    try:
        df = pd.read_excel(file_path)
        df = df.dropna()  # 移除空值
        data = df.astype(int).values  # 转换为整数数组
        return data
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


def find_best_alpha(data, state_space_1, state_space_2, param_grid):
    """寻找最佳的平滑参数alpha"""
    best_alpha = None
    best_log_likelihood = -np.inf

    for params in ParameterGrid(param_grid):
        alpha = params['alpha']
        num_states_1 = len(state_space_1)
        num_states_2 = len(state_space_2)
        transition_matrix_1 = np.full((num_states_1, num_states_1, num_states_1), alpha)
        transition_matrix_2 = np.full((num_states_2, num_states_2, num_states_2), alpha)

        for sequence in data:
            for i in range(2, len(sequence)):
                if i < 5:
                    prev_state_1 = sequence[i - 2] - 1
                    prev_state_2 = sequence[i - 1] - 1
                    next_state = sequence[i] - 1
                    transition_matrix_1[prev_state_1, prev_state_2, next_state] += 1
                else:
                    prev_state_1 = min(sequence[i - 2] - 1, num_states_2 - 1)
                    prev_state_2 = min(sequence[i - 1] - 1, num_states_2 - 1)
                    next_state = min(sequence[i] - 1, num_states_2 - 1)
                    transition_matrix_2[prev_state_1, prev_state_2, next_state] += 1

        for i in range(num_states_1):
            for j in range(num_states_1):
                if np.sum(transition_matrix_1[i, j]) > 0:
                    transition_matrix_1[i, j] = transition_matrix_1[i, j] / np.sum(transition_matrix_1[i, j])

        for i in range(num_states_2):
            for j in range(num_states_2):
                if np.sum(transition_matrix_2[i, j]) > 0:
                    transition_matrix_2[i, j] = transition_matrix_2[i, j] / np.sum(transition_matrix_2[i, j])

        log_likelihood = 0
        for sequence in data:
            seq_log_prob = 0.0
            for i in range(2, len(sequence)):
                if i < 5:
                    prev_state_1 = sequence[i - 2] - 1
                    prev_state_2 = sequence[i - 1] - 1
                    next_state = sequence[i] - 1
                    prob = transition_matrix_1[prev_state_1, prev_state_2, next_state]
                    if prob == 0:
                        prob = 1e-10
                    seq_log_prob += np.log(prob)
                else:
                    prev_state_1 = min(sequence[i - 2] - 1, num_states_2 - 1)
                    prev_state_2 = min(sequence[i - 1] - 1, num_states_2 - 1)
                    next_state = min(sequence[i] - 1, num_states_2 - 1)
                    prob = transition_matrix_2[prev_state_1, prev_state_2, next_state]
                    if prob == 0:
                        prob = 1e-10
                    seq_log_prob += np.log(prob)
            log_likelihood += seq_log_prob

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_alpha = alpha

    return best_alpha


def build_transition_matrices(data, best_alpha, state_space_1, state_space_2):
    """构建状态转移矩阵"""
    num_states_1 = len(state_space_1)
    num_states_2 = len(state_space_2)
    transition_matrix_1 = np.full((num_states_1, num_states_1, num_states_1), best_alpha)
    transition_matrix_2 = np.full((num_states_2, num_states_2, num_states_2), best_alpha)

    for sequence in data:
        for i in range(2, len(sequence)):
            if i < 5:
                prev_state_1 = sequence[i - 2] - 1
                prev_state_2 = sequence[i - 1] - 1
                next_state = sequence[i] - 1
                transition_matrix_1[prev_state_1, prev_state_2, next_state] += 1
            else:
                prev_state_1 = min(sequence[i - 2] - 1, num_states_2 - 1)
                prev_state_2 = min(sequence[i - 1] - 1, num_states_2 - 1)
                next_state = min(sequence[i] - 1, num_states_2 - 1)
                transition_matrix_2[prev_state_1, prev_state_2, next_state] += 1

    for i in range(num_states_1):
        for j in range(num_states_1):
            if np.sum(transition_matrix_1[i, j]) > 0:
                transition_matrix_1[i, j] = transition_matrix_1[i, j] / np.sum(transition_matrix_1[i, j])

    for i in range(num_states_2):
        for j in range(num_states_2):
            if np.sum(transition_matrix_2[i, j]) > 0:
                transition_matrix_2[i, j] = transition_matrix_2[i, j] / np.sum(transition_matrix_2[i, j])

    return transition_matrix_1, transition_matrix_2


def build_nn_model(data, state_space_1, state_space_2):
    """构建神经网络模型"""
    encoder = OneHotEncoder(sparse_output=False)
    X_nn = []
    y_nn = []
    for sequence in data:
        for i in range(2, len(sequence)):
            if i < 5:
                input_state = [sequence[i - 2], sequence[i - 1]]
                target = sequence[i]
                X_nn.append(input_state)
                y_nn.append(target)
            else:
                input_state = [sequence[i - 2], sequence[i - 1]]
                target = sequence[i]
                X_nn.append(input_state)
                y_nn.append(target)

    X_nn_encoded = encoder.fit_transform(np.array(X_nn))
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp.fit(X_nn_encoded, np.array(y_nn))

    return encoder, mlp


def predict_sequence(data, state_space_1, state_space_2, transition_matrix_1, transition_matrix_2, encoder, mlp):
    """生成预测序列"""
    start_combinations_1 = data[np.random.randint(0, len(data))][:2].tolist()
    start_combinations_2 = data[np.random.randint(0, len(data))][5:].tolist()

    predicted_sequence = []
    predicted_sequence.extend(start_combinations_1)

    for _ in range(5 - len(start_combinations_1)):
        available_numbers = np.array([num for num in state_space_1 if num not in predicted_sequence])
        prev_state_1 = predicted_sequence[-2] - 1
        prev_state_2 = predicted_sequence[-1] - 1
        markov_probs = np.array([transition_matrix_1[prev_state_1, prev_state_2, num-1] for num in available_numbers])
        
        if np.sum(markov_probs) == 0:
            markov_probs = np.ones_like(markov_probs) / len(markov_probs)
        else:
            markov_probs /= np.sum(markov_probs)

        input_nn = encoder.transform([[predicted_sequence[-2], predicted_sequence[-1]]])
        nn_probs = mlp.predict_proba(input_nn)[0]
        nn_probs_filtered = np.array([nn_probs[num-1] for num in available_numbers])
        
        if np.sum(nn_probs_filtered) == 0:
            nn_probs_filtered = np.ones_like(nn_probs_filtered) / len(nn_probs_filtered)
        else:
            nn_probs_filtered /= np.sum(nn_probs_filtered)

        combined_probs = 0.5 * markov_probs + 0.5 * nn_probs_filtered
        next_state_index = np.random.choice(len(available_numbers), p=combined_probs)
        predicted_sequence.append(available_numbers[next_state_index])

    predicted_sequence.extend(start_combinations_2)

    for _ in range(2 - len(start_combinations_2)):
        available_numbers = np.array([num for num in state_space_2 if num not in predicted_sequence[-2:]])
        prev_state_1 = min(predicted_sequence[-2] - 1, len(state_space_2) - 1)
        prev_state_2 = min(predicted_sequence[-1] - 1, len(state_space_2) - 1)
        markov_probs = np.array([transition_matrix_2[prev_state_1, prev_state_2, min(num-1, len(state_space_2)-1)] 
                                for num in available_numbers])
        
        if np.sum(markov_probs) == 0:
            markov_probs = np.ones_like(markov_probs) / len(markov_probs)
        else:
            markov_probs /= np.sum(markov_probs)

        input_nn = encoder.transform([[predicted_sequence[-2], predicted_sequence[-1]]])
        nn_probs = mlp.predict_proba(input_nn)[0]
        nn_probs_filtered = np.array([nn_probs[num-1] for num in available_numbers])
        
        if np.sum(nn_probs_filtered) == 0:
            nn_probs_filtered = np.ones_like(nn_probs_filtered) / len(nn_probs_filtered)
        else:
            nn_probs_filtered /= np.sum(nn_probs_filtered)

        combined_probs = 0.5 * markov_probs + 0.5 * nn_probs_filtered
        next_state_index = np.random.choice(len(available_numbers), p=combined_probs)
        predicted_sequence.append(available_numbers[next_state_index])

    return predicted_sequence


def calculate_log_probability(predicted_sequence, transition_matrix_1, transition_matrix_2, state_space_1, state_space_2):
    """计算预测序列的对数概率"""
    log_probability = 0.0
    for i in range(2, len(predicted_sequence)):
        if i < 5:
            prev_state_1 = predicted_sequence[i - 2] - 1
            prev_state_2 = predicted_sequence[i - 1] - 1
            next_state = predicted_sequence[i] - 1
            prob = transition_matrix_1[prev_state_1, prev_state_2, next_state]
            if prob == 0:
                prob = 1e-10
            log_probability += np.log(prob)
        else:
            prev_state_1 = min(predicted_sequence[i - 2] - 1, len(state_space_2) - 1)
            prev_state_2 = min(predicted_sequence[i - 1] - 1, len(state_space_2) - 1)
            next_state = min(predicted_sequence[i] - 1, len(state_space_2) - 1)
            prob = transition_matrix_2[prev_state_1, prev_state_2, next_state]
            if prob == 0:
                prob = 1e-10
            log_probability += np.log(prob)

    return log_probability


def run_prediction(_):
    """并行执行的预测函数，使用全局共享数据"""
    global _shared_data
    if _shared_data is None:
        raise ValueError("共享数据未初始化")
    
    data, state_space_1, state_space_2, transition_matrix_1, transition_matrix_2, encoder, mlp = _shared_data
    seq = predict_sequence(data, state_space_1, state_space_2, transition_matrix_1, transition_matrix_2, encoder, mlp)
    log_prob = calculate_log_probability(seq, transition_matrix_1, transition_matrix_2, state_space_1, state_space_2)
    return seq, log_prob, np.exp(log_prob)


def init_worker(data_tuple):
    """初始化工作进程的共享数据"""
    global _shared_data
    _shared_data = data_tuple


if __name__ == "__main__":
    file_path = input("请输入包含数组的 Excel 文件路径: ")
    n = int(input("请输入要运行的次数: "))
    progress_interval = int(input("请输入每运行多少次显示一次进度: "))
    
    state_space_1 = list(range(1, 36))
    state_space_2 = list(range(1, 13))
    param_grid = {'alpha': [0.01, 0.1, 1]}

    data = read_excel_data(file_path)
    if data is not None:
        print("正在训练模型...")
        
        best_alpha = find_best_alpha(data, state_space_1, state_space_2, param_grid)
        transition_matrix_1, transition_matrix_2 = build_transition_matrices(data, best_alpha, state_space_1, state_space_2)
        encoder, mlp = build_nn_model(data, state_space_1, state_space_2)

        print(f"开始运行 {n} 次预测...")
        best_sequence = None
        best_probability = 0
        best_log_probability = -np.inf

        # 准备共享数据
        shared_data = (data, state_space_1, state_space_2, transition_matrix_1, transition_matrix_2, encoder, mlp)

        print(f"已完成 0/{n} 次预测")
        
        with concurrent.futures.ProcessPoolExecutor(initializer=init_worker, initargs=(shared_data,)) as executor:
            results = list(executor.map(run_prediction, range(n)))
            
            for i, (seq, log_prob, prob) in enumerate(results):
                if i % progress_interval == 0:
                    print(f"已完成 {i}/{n} 次预测")
                if log_prob > best_log_probability:
                    best_log_probability = log_prob
                    best_probability = prob
                    best_sequence = seq
        
        print(f"已完成 {n}/{n} 次预测")
        
        # 格式化输出结果
        cleaned_sequence = [int(num) for num in best_sequence]
        probability_str = f"{best_probability:.8f}".replace("0.", ".")
        
        print("\n预测完成!")
        print(f"在 {n} 次预测中，最高概率的结果是:")
        print('预测的数组:', cleaned_sequence)
        print('概率:', probability_str)
