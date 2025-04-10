# Semantic THAB [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14906179.svg)](https://doi.org/10.5281/zenodo.14906179)

We created our dataset using an Ouster OS2-128 (Rev 7) from sequences recorded in Aschaffenburg (Germany). 
For data annotation, we used the [Point Labeler](https://github.com/jbehley/point_labeler) from [1]. 
To be consistent with [SemanticKitti](http://www.semantic-kitti.org/) [1], we have used their class definitions.

## Statistics

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

## Labeling
We use the ouster2kitty.py script to convert Ouster recordings to the kitti format. For the SLAM we use the slam provided in the Ouster API.
```bash
# First Create OSF file with ego motion
ouster-cli source .../xxx.pcap slam viz --accum-num 20 -o .../xxx.osf
# Than use OSF file to convert the recodring
python ouster2kitti.py --save_path .../kitti --osf_path .../xxx.osf --config_path .../xxx.json
```

> [!NOTE]
> We use the LEGACY config for our Ouster recodings and record at 2048@10Hz.

## Data Preperation
See [README.md](dataset/README.md) to learn about the data preperation.

## Dataloader

### Format

Each sample consists of:
- A `.bin` file containing LiDAR point cloud (`float32`, shape: N x 4)
- A `.label` file containing packed labels (`uint32`, shape: N)

Point fields:
- `x, y, z, reflectivity`

Label fields (bitwise packed):
- **semantic label**: lower 16 bits
- **instance label**: upper 16 bits

> [!NOTE]
> For Ouster sensors the spherical projection is done by the sensor itselfe. To use the data you only have to reshape to [128,2048,4] and [128,2048] for the .bin and .label files.
---

### Features

- ✅ Computes **range images** from 3D points
- ✅ Computes **surface normals**
- ✅ Tensor conversion ready for training models


---

### Class: `SemanticKitti`

```python
SemanticKitti(
    data_path: List[Tuple[str, str]],
    rotate: bool = False,
    flip: bool = False,
    id_map: Dict[int, int] = id_map
)
```
```python
range_img: Tensor [1 x H x W]
reflectivity_img: Tensor [1 x H x W]
xyz: Tensor [3 x H x W]
normals: Tensor [3 x H x W]
semantics: Tensor [1 x H x W]
``` 
### DataLoader Example
```python
from torch.utils.data import DataLoader

depth_dataset_test = SemanticKitti(data_path, rotate=False, flip=False)
dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False)

for batch in dataloader_test:
    range_img, reflectivity, xyz, normals, semantics = batch
    ...
```
## License
Creative Commons Attribution Non Commercial 4.0 International 

## References
[1]   J. Behley et al., "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 9296-9306, doi: 10.1109/ICCV.2019.00939.
