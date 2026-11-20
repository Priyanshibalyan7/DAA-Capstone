# Mini-Courier Planner (DAA Capstone, ETCCDA302)

Pipeline: Merge Sort -> 0/1 Knapsack DP -> Dijkstra -> Nearest-Neighbour TSP.

## Run
```
python courier_planner.py                      # demo on data/small_dataset.json
python courier_planner.py --capacity 20        # change van capacity
python courier_planner.py --benchmark          # tests + runtime charts (needs: pip install matplotlib)
```
Dataset format: `data/small_dataset.json` (graph adjacency + parcel list).
See `Capstone_Report.docx` for design, results, charts and screenshots.
