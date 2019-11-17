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


def order_lists(arr1, arr2):
    idx = np.argsort(arr1)[::-1]
    arr1 = np.array(arr1)[idx]
    arr2 = np.array(arr2)[idx]

    return arr1, arr2


def plot_reward(logfile, min_y, max_y, title, max_x, labels=None, output_file=None,):
    """
    logfiles separated by : are combined
    logfiles separated by , go in separate plots
    (: binds tighter than ,)
    """
    plt.clf()
    logfiles = logfile
    split_logfiles = logfiles.split(',')
    if labels:
        labels = labels.split(',')

    for j, logfile_groups in enumerate(split_logfiles):
        epoch = []
        reward = []
        test_reward_0 = []
        test_reward_1 = []
        test_reward = []
        for logfile in logfile_groups.split(':'):
            with open(logfile, 'r') as f:
                for n, line in enumerate(f):
                    if n == 0:
                        continue  # skip first line
                    line = line.strip()
                    if line == '':
                        continue
                    d = json.loads(line)
                    if max_x is not None and d['episode'] > max_x:
                        continue
                    epoch.append(int(d['episode']))
                    reward.append(float(d['avg_reward_0']))
                    test_reward_0.append(float(d['agent0_test_reward']))
                    test_reward_1.append(float(d['agent1_test_reward']))
                    if 'test_reward' in d:
                        test_reward.append(d['test_reward'])

        while len(epoch) > 200:
            new_epoch = []
            new_reward = []
            new_test_reward = []
            new_test_reward_0 = []
            new_test_reward_1 = []

            for n in range(len(epoch) // 2):
                r = (reward[n * 2] + reward[n * 2 + 1]) / 2
                e = (epoch[n * 2] + epoch[n * 2 + 1]) // 2
                new_epoch.append(e)
                new_reward.append(r)
                new_test_reward_0.append(test_reward_0[n * 2])
                new_test_reward_1.append(test_reward_1[n * 2])
                if len(test_reward) > 0:
                    rt = (test_reward[n * 2] + test_reward[n * 2 + 1]) / 2
                    new_test_reward.append(rt)
            epoch = new_epoch
            reward = new_reward
            test_reward = new_test_reward
            test_reward_0 = new_test_reward_0
            test_reward_1 = new_test_reward_1

        if min_y is None:
            min_y = 0
        if max_y is not None:
            plt.ylim([min_y, max_y])
        suffix = ''
        if len(split_logfiles) > 0:
            suffix = ' %s' % (j + 1)
        if len(test_reward) > 0:
            label = labels[j] + ' ' if labels else ''
            plt.plot(np.array(epoch) / 1000, reward, label=label + 'joint train' + suffix)
            plt.plot(np.array(epoch) / 1000, test_reward, label=label + 'joint test' + suffix)
            plt.plot(np.array(epoch) / 1000, test_reward_0, label=label + 'Agent A test' + suffix)
            plt.plot(np.array(epoch) / 1000, test_reward_1, label=label + 'Agent B test' + suffix)

        else:
            plt.plot(np.array(epoch) / 1000, reward, label='reward' + suffix)
    if title is not None:
        plt.title(title)
    plt.xlabel('Episodes of 128 games (thousands)')
    plt.ylabel('Reward')
    plt.legend()
    if output_file:
        plt.savefig(output_file)


def plot_training_curve(filename, min_y=0, max_y=1, title='', max_x=200000, labels=None, output=None):
    """

    """
    plt.clf()
    epoch = []
    test_reward_0 = []
    test_reward_1 = []
    with open(filename, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                continue  # skip first line
            line = line.strip()
            if line == '':
                continue
            d = json.loads(line)
            if max_x is not None and d['episode'] > max_x:
                continue
            epoch.append(int(d['episode']))
            test_reward_0.append(float(d['agent0_test_reward']))
            test_reward_1.append(float(d['agent1_test_reward']))

        while len(epoch) > 200:
            new_epoch = []
            new_test_reward_0 = []
            new_test_reward_1 = []

            for n in range(len(epoch) // 2):
                e = (epoch[n * 2] + epoch[n * 2 + 1]) // 2
                new_epoch.append(e)
                new_test_reward_0.append(test_reward_0[n * 2])
                new_test_reward_1.append(test_reward_1[n * 2])

            epoch = new_epoch
            test_reward_0 = new_test_reward_0
            test_reward_1 = new_test_reward_1

        if min_y is None:
            min_y = 0
        if max_y is not None:
            plt.ylim([min_y, max_y])
        suffix = ''
        plt.plot(np.array(epoch) / 1000, test_reward_0, label='test 0' + suffix)
        plt.plot(np.array(epoch) / 1000, test_reward_1, label='test 1' + suffix)

    if title is not None:
        plt.title(title)
    plt.xlabel('Episodes of 128 games (thousands)')
    plt.ylabel('Reward')
    plt.legend()
    if output:
        plt.savefig(output)
