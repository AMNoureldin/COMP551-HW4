import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from argparse import ArgumentParser
from os import listdir
from pathlib import Path

MLP_PAT = r'^(?P<dataset>[A-Za-z\d]+)_(?P<model>[A-Za-z\d]+)_.*_base_(?P<width>\d+)_width_(?P<max_width>\d+).*$'
RES_PAT = r'^(?P<dataset>[A-Za-z\d]+)_(?P<model>[A-Za-z\d]+)_(?P<width>\d+)_(?P<max_width>\d+).*$'

def load_data(directory, pat):
    dirs = listdir(directory)
    result = {}
    resultdf = pd.DataFrame({'model':[],
                             'dataset':[],
                             'width':[],
                             'max_width': [],
                             'train_acc': [],
                             'train_loss': [],
                             'test_acc':[],
                             'test_loss':[]})
    for i, x in enumerate(dirs):
        print(x)
        pat_match = re.search(pat, x)
        if not pat_match:
            continue
        cur_run: dict = pat_match.groupdict()
        path = f'{directory}/{x}'

        try:
            acc_test = pd.read_csv(path + '/acc_test.csv').iloc[0].values
            acc_test = np.delete(acc_test, [0, 1])
            acc_test = truncate(acc_test)
            acc_train = pd.read_csv(path + '/acc_train.csv').iloc[0].values
            acc_train = np.delete(acc_train, [0, 1])
            acc_train = truncate(acc_train)
            loss_test = pd.read_csv(path + '/loss_test.csv').iloc[0].values
            loss_test = np.delete(loss_test, [0, 1])
            loss_test = truncate(loss_test)
            loss_train = pd.read_csv(path + '/loss_train.csv').iloc[0].values
            loss_train = np.delete(loss_train, [0, 1])
            loss_train = truncate(loss_train)
            df = pd.DataFrame({**cur_run, 'train_loss': [loss_train],
                               'train_acc': [acc_train], 'test_acc': [acc_test], 'test_loss': [loss_test]})
            cur_run['train'] = {'acc': acc_train, 'loss': loss_train}
            cur_run['test'] = {'acc': acc_test, 'loss': loss_test}
            result[str(i)] = cur_run

            resultdf = resultdf.append(df, ignore_index=True)

        except FileNotFoundError:
            try:
                acc_test = pd.read_csv(path + '/acc1_test.csv').iloc[0].values
                acc_test = np.delete(acc_test, [0, 1])
                acc_test = truncate(acc_test)
                acc_train = pd.read_csv(path + '/acc1_train.csv').iloc[0].values
                acc_train = np.delete(acc_train, [0, 1])
                acc_train = truncate(acc_train)
                loss_test = pd.read_csv(path + '/loss_test.csv').iloc[0].values
                loss_test = np.delete(loss_test, [0, 1])
                loss_test = truncate(loss_test)
                loss_train = pd.read_csv(path + '/loss_train.csv').iloc[0].values
                loss_train = np.delete(loss_train, [0, 1])
                loss_train = truncate(loss_train)
                df = pd.DataFrame({**cur_run, 'train_loss': [loss_train],
                                   'train_acc': [acc_train], 'test_acc': [acc_test], 'test_loss': [loss_test]})
                cur_run['train'] = {'acc': acc_train, 'loss': loss_train}
                cur_run['test'] = {'acc': acc_test, 'loss': loss_test}
                result[str(i)] = cur_run

                resultdf = resultdf.append(df, ignore_index=True)
            except FileNotFoundError:
                print(f'Error fetching data from {path}....')

    return result, resultdf

def truncate(arr):
    if len(arr) > 300:
        return arr[len(arr) - 300:]
    else:
        return arr
    
def plot_simple(data, out_dir):
    out_path = Path(out_dir)
    for _, v in data.items():
        model = v['model']
        dataset = v['dataset']
        title = f"{model}_{dataset}_{v['width']}_{v['max_width']}"
        run_dir = out_path / f'{model}/{dataset}/{title}'
        run_dir.mkdir(parents=True, exist_ok=True)
        train_ = v['train']
        test_ = v['test']
        if len(train_['acc']) > len(test_['acc']):
            train_['acc'] = train_['acc'][:len(test_['acc'])]
        elif len(train_['acc']) < len(test_['acc']):
            test_['acc'] = test_['acc'][:len(train_['acc'])]
        if len(train_['loss']) > len(test_['loss']):
            train_['loss'] = train_['loss'][:len(test_['loss'])]
        elif len(train_['loss']) < len(test_['acc']):
            test_['loss'] = test_['loss'][:len(train_['loss'])]
        x = range(1, min(len(train_['acc']), len(test_['acc'])) + 1)
        acc = get_two_y_graph(x, train_['acc'], test_['acc'], title)
        loss = get_two_y_graph(x, train_['acc'], test_['acc'], title)
        combine = get_split_acc_loss(x, (train_['acc'], test_['acc']), (train_['loss'], test_['loss']), title)
        acc.savefig(str(run_dir / 'accuracy.png'))
        loss.savefig(str(run_dir / 'loss.png'))
        combine.savefig(str(run_dir / 'combined.png'))
        
def get_two_y_graph(x, y_trn, y_tst, title, ylabel='Accuracy (%)', xlabel='Epoch'):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(x, y_tst, '-o', label='Testing')
    ax.plot(x, y_trn, '-o', label='Training')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.legend(loc='center right')
    fig.set_facecolor('lightsteelblue')
    return fig

def get_split_acc_loss(x, y_acc, y_loss, title,  ylabel='Accuracy (%)', xlabel='Epoch'):
    y_trn, y_tst = y_acc
    ly_trn, ly_tst = y_loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
    ax1.plot(x, y_tst, '-o', label='Testing')
    ax1.plot(x, y_trn, '-o', label='Training')
    ax2.plot(x, ly_tst, '-o', label='Testing')
    ax2.plot(x, ly_trn, '-o', label='Training')
    fig.supxlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel('Loss')
    fig.suptitle(title)
    fig.legend(loc='center right')
    fig.set_facecolor('lightsteelblue')
    return fig

def clean_df(df):
    #print(df)
    df['max_trn_acc'] = df.apply(lambda row: np.max(row['train_acc']), axis=1)
    df['max_tst_acc'] = df.apply(lambda row: np.max(row['test_acc']), axis=1)
    df['min_trn_loss'] = df.apply(lambda row: np.max(row['train_loss']), axis=1)
    df['min_tst_loss'] = df.apply(lambda row: np.max(row['test_loss']), axis=1)
    df['widening'] = df.apply(lambda row: float(row.max_width)/ float(row.width), axis=1)
    df['connectivity'] = df.apply(lambda row: int(row['width'])/int(row['max_width']), axis=1)
    df['width'] = pd.to_numeric(df['width'])
    df['max_width'] = pd.to_numeric(df['max_width'])
    df = df.sort_values('max_width')
    return df

def plot_res_net_comb(df, out_dir):
    grouped = df.groupby('dataset')
    cifar10 = grouped.get_group('cifar10')
    cifar100 = grouped.get_group('cifar100')
    svhn_group = grouped.get_group('svhn')
    s_width_grps = svhn_group.groupby('width')
    c10_width_grps = cifar10.groupby('width')
    c100_width_grps = cifar100.groupby('width')
    c100_y1 = c100_width_grps.get_group(8)
    c100_y2 = c100_width_grps.get_group(18)
    c10_y1 = c10_width_grps.get_group(8)
    c10_y2 = c10_width_grps.get_group(18)
    s_y1 = s_width_grps.get_group(8)
    s_y2 = s_width_grps.get_group(18)
    # =========== Training Graph ==============
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7,18), constrained_layout=True)
    ax2.plot(c100_y1['widening'], c100_y1['max_trn_acc'], 'o', label='1.8e+05')
    ax2.plot(c100_y2['widening'], c100_y2['max_trn_acc'], 'o', label='9.0e+05')
    ax1.plot(c10_y1['widening'], c10_y1['max_trn_acc'], 'o', label='1.8e+05')
    ax1.plot(c10_y2['widening'], c10_y2['max_trn_acc'], 'o', label='9.0e+05')
    ax3.plot(s_y1['widening'], s_y1['max_trn_acc'], 'o', label='1.8e+05')
    ax3.plot(s_y2['widening'], s_y2['max_trn_acc'], 'o', label='9.0e+05')
    ax1.set_xscale('log')
    ax1.set_xlabel('Widening Factor')
    ax1.set_ylabel('Accuracy (in %)')
    ax1.set_title('CIFAR 10')
    ax2.set_xscale('log')
    ax2.set_xlabel('Widening Factor')
    ax2.set_ylabel('Accuracy (in %)')
    ax2.set_title('CIFAR 100')
    ax3.set_xscale('log')
    ax3.set_xlabel('Widening Factor')
    ax3.set_ylabel('Accuracy (in %)')
    ax3.set_title('SVHN')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    fig.suptitle('Training Results')
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.savefig(f'{out_dir}/resnet_combined_train.png')
    # =========== Testing Graph ==============
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7,18), constrained_layout=True)
    ax2.plot(c100_y1['widening'], c100_y1['max_tst_acc'], 'o', label='1.8e+05')
    ax2.plot(c100_y2['widening'], c100_y2['max_tst_acc'], 'o', label='9.0e+05')
    ax1.plot(c10_y1['widening'], c10_y1['max_tst_acc'], 'o', label='1.8e+05')
    ax1.plot(c10_y2['widening'], c10_y2['max_tst_acc'], 'o', label='9.0e+05')
    ax3.plot(s_y1['widening'], s_y1['max_tst_acc'], 'o', label='1.8e+05')
    ax3.plot(s_y2['widening'], s_y2['max_tst_acc'], 'o', label='9.0e+05')
    ax1.set_xscale('log')
    ax1.set_xlabel('Widening Factor')
    ax1.set_ylabel('Accuracy (in %)')
    ax1.set_title('CIFAR 10')
    ax2.set_xscale('log')
    ax2.set_xlabel('Widening Factor')
    ax2.set_ylabel('Accuracy (in %)')
    ax2.set_title('CIFAR 100')
    ax3.set_xscale('log')
    ax3.set_xlabel('Widening Factor')
    ax3.set_ylabel('Accuracy (in %)')
    ax3.set_title('SVHN')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    fig.suptitle('Testing Results')
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.savefig(f'{out_dir}/resnet_combined_test.png')

def plot_mlp_comb(df, out_dir):
    dense = df.loc[df['width'] == df['max_width']]
    sparse = df.loc[df['width'] != df['max_width']]
    # =========== Test Accuracy ============
    fig, ax = plt.subplots(figsize=(9,6), constrained_layout=True)
    ax.plot(dense['width'], dense['max_tst_acc'], '-x', label='dense')
    ax.plot(sparse['max_width'], sparse['max_tst_acc'], '-x', label='sparse')
    fig.suptitle('MLP Test Accuracy')
    fig.legend()
    ax.grid()
    ax.set_xlabel('Width')
    ax.set_ylabel('Accuracy')
    #ax.set_ylim([0.9, 1.05])
    fig.savefig(f'{out_dir}/MLP_width_acc_tst.png')
    # =========== Train Accuracy ============
    fig, ax = plt.subplots(figsize=(9,6), constrained_layout=True)
    ax.plot(dense['width'], dense['max_trn_acc'], '-x', label='dense')
    ax.plot(sparse['max_width'], sparse['max_trn_acc'], '-x', label='sparse')
    fig.suptitle('MLP Train Accuracy')
    fig.legend()
    ax.grid()
    ax.set_xlabel('Width')
    ax.set_ylabel('Accuracy')
    #ax.set_ylim([0.9, 1.05])
    fig.savefig(f'{out_dir}/MLP_width_acc_trn.png')
    # =========== Testing Loss ============
    fig, ax = plt.subplots(figsize=(9,6), constrained_layout=True)
    ax.plot(dense['width'], dense['min_tst_loss'], '-x', label='dense')
    ax.plot(sparse['max_width'], sparse['min_tst_loss'], '-x', label='sparse')
    fig.suptitle('MLP Test Loss')
    fig.legend()
    ax.grid()
    #ax.set_ylim([0.9, 1.05])
    ax.set_xlabel('Width')
    ax.set_ylabel('Accuracy')
    #ax.set_ylim([0.9, 1.05])
    fig.savefig(f'{out_dir}/MLP_width_loss_tst.png')
    # =========== Training Loss ============
    fig, ax = plt.subplots(figsize=(9,6), constrained_layout=True)
    ax.plot(dense['width'], dense['min_trn_loss'], '-x', label='dense')
    ax.plot(sparse['max_width'], sparse['min_trn_loss'], '-x', label='sparse')
    fig.suptitle('MLP Train Loss')
    fig.legend()
    ax.grid()
    #ax.set_ylim([0.9, 1.05])
    ax.set_xlabel('Width')
    ax.set_ylabel('Accuracy')
    #ax.set_ylim([0.9, 1.05])
    fig.savefig(f'{out_dir}/MLP_width_loss_trn.png')
    

if __name__ == '__main__':
    parser = ArgumentParser(prog='graph_results')
    parser.add_argument('-o', '--out_dir', required=False, default='results', type=str,
                        help="Output directory to put results in. Format: path/to/dir (no trailing '/')")
    parser.add_argument('-m', '--model', required=False, default='All', type=str,
                        help="Extract results from multiple runs in 'in_dir' directory.")
    args = parser.parse_args()
    out_dir = args.out_dir
    model = args.model
    if model == 'ResNet18':
        datadict, df = load_data('results/ResNet18', RES_PAT)
        df = clean_df(df)
        plot_simple(datadict, out_dir)
        plot_res_net_comb(df, out_dir)
    elif model == 'MLP':
        datadict, df = load_data('results/MLP', MLP_PAT)
        plot_simple(datadict, out_dir)
        df = clean_df(df)
        plot_mlp_comb(df, out_dir)
    else:
        datadictMLP, dfMLP = load_data('results/MLP', MLP_PAT)
        dfMLP = clean_df(dfMLP)
        plot_simple(datadictMLP, out_dir)
        datadictRes, dfRes = load_data('results/ResNet18', RES_PAT)
        dfRes = clean_df(dfRes)
        plot_simple(datadictRes, out_dir)
        plot_res_net_comb(dfRes, out_dir)
        plot_mlp_comb(dfMLP, out_dir)
