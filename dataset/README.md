# Semantic LiDAR Dataset Utils

### Semantic Kitti
Download the [SemanticKitti](http://www.semantic-kitti.org/) dataset [1].

Extract the folders to ./dataset

Ensure the following data structure:

```
├── data
│   ├── SemanticKitti
│   │   ├── dataset
│   │   │   ├── sequences
│   │   │   │   ├── 00
│   │   │   │   │   ├── velodyne
│   │   │   │   │   │   ├── *.bin
│   │   │   │   │   ├── label
│   │   │   │   │   │   ├── *.label
```


### Semantic THAB [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14906179.svg)](https://doi.org/10.5281/zenodo.14906179)

We created our dataset using an Ouster OS2-128 (Rev 7) from sequences recorded in Aschaffenburg (Germany). 
For data annotation, we used the [Point Labeler](https://github.com/jbehley/point_labeler) from [1]. 
To be consistent with [SemanticKitti](http://www.semantic-kitti.org/) [1], we have used their class definitions.


| Date | Sequences |  Status    | Size | Meta | Note
|:----:|:---------:|:-------------:|:---------:|:------:|:------:|
| 070324    | 0000 | $${\color{green}Online}$$ |  1090  | Residential Area / Industrial Area |
| 190324    | 0001 | $${\color{green}Online}$$ |  344   | City Ring Road                     |
| 190324    | 0002 | $${\color{green}Online}$$ |  228   | Inner City                         |
| 190324    | 0003 | $${\color{green}Online}$$ |  743   | Pedestrian Area                    | Fixed Lane Markings in v3
| 190324    | 0004 | $${\color{green}Online}$$  |  400   | Inner City                         |
| 190324    | 0005 | $${\color{green}Online}$$  |  603   | Inner City                         |
| 190324    | 0006 | $${\color{green}Online}$$  |  320   | Inner City                         |
| 190324    | 0007 | $${\color{green}Online}$$  |  517   | Residential Area & Campus TH AB    | 
| 190324    | 0008 | $${\color{green}Online}$$  |  505   | Campus TH AB                       | Added Labels in v3

> [!NOTE]
> For Ouster sensors the spherical projection is done by the sensor itselfe. To use the data you only have to reshape to [128,2048,4] and [128,2048] for the .bin and .label files.

Ensure the following data structure:

```
├── data
│   ├── SemanticTHAB
│   │   ├── sequences 
│   │   │   ├── 0001 # SequenceID
│   │   │   │   ├── velodyne
│   │   │   │   │   ├── *.bin
│   │   │   │   ├── label
│   │   │   │   │   ├── *.label

```
<a name="license"></a>

### References
[1]   J. Behley et al., "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 9296-9306, doi: 10.1109/ICCV.2019.00939.


