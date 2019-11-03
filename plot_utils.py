import json
import matplotlib.pyplot as plt
import numpy as np


def extract_measures(filename, measures=[]):
    data = {measure: [] for measure in measures}

    with open(filename, 'r') as f:
        for n, line in enumerate(f):
                    if n == 0:
                        continue  # skip first line
                    line = line.strip()
                    if line == '':
                        continue
                    d = json.loads(line)
                    for measure in measures:
                        data[measure].append(float(d[measure]))
    return data


def load_logs(filename):
    epoch = []
    reward = []
    test_reward = []
    test_utterance = []
    with open(filename, 'r') as f:
        for n, line in enumerate(f):
                    if n == 0:
                        continue  # skip first line
                    line = line.strip()
                    if line == '':
                        continue
                    d = json.loads(line)
                    epoch.append(int(d['episode']))
                    reward.append(float(d['avg_reward_0']))
                    if 'test_reward' in d:
                        test_reward.append(d['test_reward'])
                    test_utterance.append(d['test_symbols_used'])
    return epoch, reward, test_reward, test_utterance


def extract_data(filename):
    epoch, reward, test_reward, test_utterance = load_logs(filename)

    test_utterance_per_position = [[{} for _ in range(6)] for _ in range(128)]
    for epoch in test_utterance:
        for trajectory in epoch:
            for i, utterance in enumerate(trajectory):
                for j, symbol in enumerate(utterance):
                    symbol = str(symbol)
                    if symbol in test_utterance_per_position[i][j]:
                        test_utterance_per_position[i][j][symbol] += 1
                    else:
                        test_utterance_per_position[i][j][symbol] = 1
    return test_utterance_per_position


def extract_utterance(filename):
    messages = [[] for _ in range(10)]

    with open(filename, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                continue
            if line == '':
                continue
            d = json.loads(line)
            symbols_used = d['test_symbols_used']
            for i, msg in enumerate(symbols_used):
                messages[i] += msg
    return messages


def get_distributions(messages, n=3, msg_len=6, vocab_size=10):
    distributions = []
    for i in range(n):
        messages_in_position = messages[i]
        messages_in_position = np.array([np.array(msg) for msg in messages_in_position])
        msg_distribution = [[0 for _ in range(vocab_size)] for _ in range(msg_len)]
        for position in range(msg_len):
            unique, counts = np.unique(messages_in_position[:, position], return_counts=True)
            for u, c in zip(unique, counts):
                msg_distribution[position][u] = c
        distributions.append(msg_distribution)

    distributions = np.array(distributions)
    return distributions


def get_x_positions(inner_no, outer_no, width=2, inner_width=0.2, outer_width=0.5):
    """
    something is wrong with the way its interpreted by matplotlib
    """
    positions = []
    last = 0
    for i in range(outer_no):

        last += outer_width
        inner_positions = [last]

        for j in range(inner_no - 1):
            last += width + inner_width
            inner_positions.append(last)

        last += width + outer_width
        positions.append(np.array(inner_positions))
    return np.array(positions)
