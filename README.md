# Semantic LiDAR training tool 

Tool for semantic lidar model training.

[![Watch the video](https://cdn.discordapp.com/attachments/709432890458374204/1219546130115727390/image.png?ex=66309bd7&is=661e26d7&hm=c48cbefebdc49abcba54b0350bd200d4fae5accf0a629c695a429e82c0eac7f9&)](https://drive.google.com/file/d/1R7l4302yjyHZzcCP7Cm9vKr7sSnPDih_/view)
## Development environment:

### VS-Code:
The project is designed to be delevoped within vs-code IDE using remote container development.

## Usage:
### Setup Docker Container
In docker-compse.yaml all parameters are defined.
```bash
# Enable xhost in the terminal
sudo xhost +

# Add user to environment
sh setup.sh

# Build the image from scratch using Dockerfile, can be skipped if image already exists or is loaded from docker registry
docker-compose build --no-cache

# Start the container
docker-compose up -d

# Stop the container
docker compose down
```
### Train Semantic Kitti
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

### Train Semantic THAB
We created our dataset using an Ouster OS2-128 (Rev 7) from sequences recorded in Aschaffenburg (Germany). 
For data annotation, we used the [Point Labeler](https://github.com/jbehley/point_labeler) from [1]. 
To be consistent with [SemanticKitti](http://www.semantic-kitti.org/) [1], we have used their class definitions.


| Date | Sequences |  Status    | Size | Meta | Split
|:----:|:---------:|:-------------:|:---------:|:------:|:------:|
| 070324    | [[0001]](https://drive.google.com/file/d/1HY0PkAwEnBLZRL1_-m4tZtezJxW2Pu5Y/view?usp=sharing)    | $${\color{green}Online}$$ |  1090  | Residential Area / Industrial Area | Train
| 190324    | [[0001]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{green}Online}$$ |  344   | City Ring Road                     | Train
| 190324    | [[0002]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{green}Online}$$ |  228   | Inner City                         | Eval
| 190324    | [[0003]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{green}Online}$$ |  743   | Pedestrian Area                    | Train
| 190324    | [[0004]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{red}Oflline}$$  |  400   | Inner City                         | Train
| 190324    | [[0005]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{red}Oflline}$$  |  603   | Inner City                         | Test
| 190324    | [[0006]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{red}Oflline}$$  |  ??   | Inner City                          | Test
| 190324    | [[0007]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{red}Oflline}$$  |  ??   | Residential Area & Campus TH AB     | Test
| 190324    | [[0008]](https://drive.google.com/file/d/1Uvms0DD5SUiGKAVgz6WQZWCwCnaOdUBd/view?usp=sharing)    | $${\color{red}Oflline}$$  |  ??   | Campus TH AB                        | Train

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
### Model Zoo

#### SemanticKitti
| Backbone | Parameters | Inference Time¹ | mIoU² | Status | Revision³
|:--------:|:----------:|:---------------:|:----:|:------:|:------:|
| [[ResNet18]](https://drive.google.com/file/d/1HY0PkAwEnBLZRL1_-m4tZtezJxW2Pu5Y/view?usp=sharing) |  18 M      |  10ms  | 49%  | $${\color{green}Online}$$ | 0.0.1
| [[ResNet34]](https://drive.google.com/file/d/1HY0PkAwEnBLZRL1_-m4tZtezJxW2Pu5Y/view?usp=sharing) |  28 M      |  14ms  | 54%  | $${\color{green}Online}$$ | 0.0.1
| [[ResNet50]](https://drive.google.com/file/d/1HY0PkAwEnBLZRL1_-m4tZtezJxW2Pu5Y/view?usp=sharing) |  128 M     |  50ms  | 54.5%  | $${\color{green}Online}$$ | 0.0.1

#### SemanticTHAB
| Backbone | Parameters | Inference Time¹ | mIoU² | Status | Revision³
|:--------:|:----------:|:---------------:|:----:|:------:|:------:|
| [[ResNet18]](https://drive.google.com/file/d/1HY0PkAwEnBLZRL1_-m4tZtezJxW2Pu5Y/view?usp=sharing) |  18 M      |  10ms  | 49%  | $${\color{green}Online}$$ | 0.0.1
| [[ResNet34]](https://drive.google.com/file/d/1HY0PkAwEnBLZRL1_-m4tZtezJxW2Pu5Y/view?usp=sharing) |  28 M      |  14ms  | 54%  | $${\color{green}Online}$$ | 0.0.1
| [[ResNet50]](https://drive.google.com/file/d/1HY0PkAwEnBLZRL1_-m4tZtezJxW2Pu5Y/view?usp=sharing) |  128 M     |  50ms  | 54.5%  | $${\color{green}Online}$$ | 0.0.1

¹ Inference time measured at a Nivida Geforce RTX 2070 TI.

² Model input size is 128x2048, mIoU is measured over the Eval set.

³ See the revision log for details on changes in architecture and training procedure.

### References
[1]   J. Behley et al., "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 9296-9306, doi: 10.1109/ICCV.2019.00939.


