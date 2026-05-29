# Raw Data

Raw data files are not committed to this repo — they're too large and sourced externally.

## ACN-Data

Export session JSON files from [acndata.caltech.edu](https://acndata.caltech.edu/) and place them inside `data/raw/acn/`.

## ST-EVCDP

Clone or download [IntelligentSystemsLab/ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP) and copy the `datasets/` folder contents into `data/raw/urbanev/`. The pipeline expects:

```
information.csv   price.csv   time.csv
occupancy.csv     volume.csv  duration.csv
```
