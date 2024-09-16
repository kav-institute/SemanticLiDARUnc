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


### Semantic THAB

We created our dataset using an Ouster OS2-128 (Rev 7) from sequences recorded in Aschaffenburg (Germany). 
For data annotation, we used the [Point Labeler](https://github.com/jbehley/point_labeler) from [1]. 
To be consistent with [SemanticKitti](http://www.semantic-kitti.org/) [1], we have used their class definitions.


| Date | Sequences |  Status    | Size | Meta | Split
|:----:|:---------:|:-------------:|:---------:|:------:|:------:|
| 070324    | [[0001]](https://drive.google.com/file/d/1v6ChrQ8eaOKVz2kEZmVoTz3aY2B46eN6/view?usp=sharing)    | $${\color{green}Online}$$ |  1090  | Residential Area / Industrial Area | Train
| 190324    | [[0001]](https://drive.google.com/file/d/1I69_bAd4E_1VeGDvnlf2HgxgVJnEhc3G/view?usp=sharing)    | $${\color{green}Online}$$ |  344   | City Ring Road                     | Train
| 190324    | [[0002]](https://drive.google.com/file/d/1fJ2uhToOQArDZW0wQcnDWeLQViExk7Zy/view?usp=sharing)    | $${\color{green}Online}$$ |  228   | Inner City                         | Train
| 190324    | [[0003]](https://drive.google.com/file/d/167E8YQWMhifcUOtMSgp-YpCiEAR72gJA/view?usp=sharing)    | $${\color{green}Online}$$ |  743   | Pedestrian Area                    | Train
| 190324    | [[0004]](https://de.wikipedia.org/wiki/HTTP_404)    | $${\color{red}Offline}$$  |  400   | Inner City                         | Train
| 190324    | [[0005]](https://de.wikipedia.org/wiki/HTTP_404)    | $${\color{red}Offline}$$  |  603   | Inner City                         | Test
| 190324    | [[0006]](https://de.wikipedia.org/wiki/HTTP_404)    | $${\color{red}Offline}$$  |  320   | Inner City                          | Test
| 190324    | [[0007]](https://de.wikipedia.org/wiki/HTTP_404)    | $${\color{red}Offline}$$  |  517   | Residential Area & Campus TH AB     | Test
| 190324    | [[0008]](https://de.wikipedia.org/wiki/HTTP_404)    | $${\color{red}Offline}$$  |  505   | Campus TH AB                        | Train

> [!NOTE]
> For Ouster sensors the spherical projection is done by the sensor itselfe. To use the data you only have to reshape to [128,2048,4] and [128,2048] for the .bin and .label files.

Ensure the following data structure:

```
├── data
│   ├── SemanticTHAB
│   │   ├── 070323 # Date
│   │   │   ├── 0001 # SequenceID
│   │   │   │   ├── velodyne
│   │   │   │   │   ├── *.bin
│   │   │   │   ├── label
│   │   │   │   │   ├── *.label

```
<a name="license"></a>

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


## License:
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

### References
[1]   J. Behley et al., "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 9296-9306, doi: 10.1109/ICCV.2019.00939.


