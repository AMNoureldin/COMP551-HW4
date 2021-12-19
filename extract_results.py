import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath, out_dir=''):

    final_out = {}
    for dname in os.listdir(dpath):
        print(f"Converting run {dname}",end="")
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        tags = ea.Tags()['scalars']

        out = {}

        for tag in tags:
            tag_values=[]
            wall_time=[]
            steps=[]

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])

        if len(tags)>0:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            df= pd.concat(out.values(),keys=out.keys())
            df.to_csv(f'{out_dir}/{dname}.csv')
            print("- Done")
            final_out[dname] = df
        else:
            print('- Not scalers to write')




    return final_out


def tabulate_multi(dpath, out_dir=''):
    in_prnt = Path(dpath)
    out_prnt = Path(out_dir)
    out_prnt.mkdir(parents=True, exist_ok=True)
    for dname in os.listdir(dpath):
        print(f'Tabulating run: {dname}...')
        in_path = in_prnt / dname
        out_path = out_prnt / dname
        out_path.mkdir()
        steps = tabulate_events(str(in_path), str(out_path))
        if len(steps.values()) > 0:
            # print(steps)
            pd.concat(steps.values(), keys=steps.keys()).to_csv(f'{dpath}/{dname}_all_result.csv')
        print('Done.')


if __name__ == '__main__':
    parser = ArgumentParser(prog='extract_results')
    parser.add_argument('-o', '--out_dir', required=False, default='results', type=str,
                        help="Output directory to put results in. Format: path/to/dir (no trailing '/')")
    parser.add_argument('-i', '--in_dir', required=False, default='runs', type=str,
                        help="Input directory to get results from.")
    parser.add_argument('-m', '--multi', required=False, default=False, action='store_true',
                        help="Extract results from multiple runs in 'in_dir' directory.")
    args = parser.parse_args()
    path = args.in_dir
    out_dir = args.out_dir
    if args.multi:
        tabulate_multi(path, out_dir)
    else:
        steps = tabulate_events(path, out_dir)
        
        if len(steps.values()) > 0:
            # print(steps)
            pd.concat(steps.values(),keys=steps.keys()).to_csv(f'{out_dir}/all_result.csv')
