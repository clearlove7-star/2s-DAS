import os
import numpy as np

root_dir = './result'


def summary_results(prefix, opts, repeats, splits, epochs, mode, train_test='test'):
    all_results = np.zeros((len(opts), len(repeats), len(splits), len(epochs), 5))

    missing_count = 0
    for opt_id, opt in enumerate(opts):
        for repeat_id, repeat in enumerate(repeats):
            for split_id, split in enumerate(splits):

                if opt is None:
                    result_dir = os.path.join(root_dir, f'{prefix}-S{split}-{repeat}')
                else:
                    result_dir = os.path.join(root_dir, f'{prefix}-S{split}-{repeat}-{opt_id}')

                for epoch_id, epoch in enumerate(epochs):

                    result_file = os.path.join(result_dir, f'{train_test}_results_{mode}_epoch{epoch}.npy')
                    print(result_file)

                    try:
                        result = np.load(result_file, allow_pickle=True).item()
                    except:
                        all_results[opt_id, repeat_id, split_id, epoch_id] = all_results[
                            opt_id, repeat_id, split_id, epoch_id - 1]
                        missing_count += 1
                        continue

                    all_results[opt_id, repeat_id, split_id, epoch_id, 0] = result['Acc']
                    all_results[opt_id, repeat_id, split_id, epoch_id, 1] = result['Edit']
                    all_results[opt_id, repeat_id, split_id, epoch_id, 2] = result['F1@10']
                    all_results[opt_id, repeat_id, split_id, epoch_id, 3] = result['F1@25']
                    all_results[opt_id, repeat_id, split_id, epoch_id, 4] = result['F1@50']

    print(f'Missing: {missing_count}')

    return all_results


def print_all_results(all_results, opts, repeats, splits, epochs):
    for opt_id, opt in enumerate(opts):
        for repeat_id, repeat in enumerate(repeats):
            for split_id, split in enumerate(splits):
                print(f'\nOptions: {opt}, Repeat: {repeat}, Split: {split}')
                for epoch_id, epoch in enumerate(epochs):
                    results = all_results[opt_id, repeat_id, split_id, epoch_id]
                    print(f'Epoch {epoch}: Acc={results[0]:.1f}, Edit={results[1]:.1f}, F1@10={results[2]:.1f}, F1@25={results[3]:.1f}, F1@50={results[4]:.1f}')
                print('-------------')

def get_best_epochs(results, epochs, window_size):
    # results: (epoch_num, )

    max_value = 0
    max_index = -1
    for o in range(len(epochs) - window_size + 1):
        if results[o:o + window_size].mean() > max_value:
            max_value = results[o:o + window_size].mean()
            max_index = o

    return epochs[max_index:max_index + window_size]


prefix = 'GTEA-Trained'

opts = [None]

splits = [4]
epochs = [i for i in range(0, 10001, 50)]
window_size = 1
mode = 'decoder-agg'
repeats = [0]

all_results = summary_results(prefix, opts, repeats, splits, epochs, mode, train_test='test')

print_all_results(all_results, opts, repeats, splits, epochs)

for opt_id, opt in enumerate(opts):
    best_epochs = get_best_epochs(all_results[opt_id].mean(0).mean(0).mean(1), epochs, window_size)
    best_epoch_ids = [epochs.index(i) for i in best_epochs]
    # print(best_epoch_ids)

    results = all_results[opt_id].mean(0).mean(0)[best_epoch_ids].mean(0)

    print(best_epochs, results.mean())

    print('Opt', opt)
    print('-------------')
    print(results.mean())
    print('-------------')
    print(results)
    print('-------------')
    print(
        f'multicolumn{{3}}{{c}}{{ {results[2]:.1f} / {results[3]:.1f} / {results[4]:.1f} }} & {results[1]:.1f} & {results[0]:.1f} & {results.mean():.1f}')
    print('-------------')

    print()
    print()