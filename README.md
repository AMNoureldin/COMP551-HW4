# Report on the reproduction of "Are wider nets better given the same number of parameters?"
---

### Reproducing results:
CLI arguments used to reproduce the results of the paper can be found under `<model>/args.txt`. Each line represents one fit for one dataset. Commands should be run from the main working directory of the project.

### Extracting CSV results:

Run the following commands from the working directory of the project
```
python extract_results.py -i MLP/runs -o results/MLP -m
python extract_results.py -i ResNet18/runs -o results/ResNet18 -m
```
### Producing graphs:

To produce the graphs shown in the report along with the graphs for each fit, run the following code after extracting the csv results. You will find graphs under results/graph

```
python graph_results.py -o results/graphs
```
---
Repository for project can be found at https://github.com/AMNoureldin/COMP551-HW4
