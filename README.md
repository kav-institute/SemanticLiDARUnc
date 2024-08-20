# Semantic LiDAR

A tool for training and finetuning of a semantic segmentation model on data of an Ouster OS2-128 (Rev 7), collected @ TH AB

[![Watch the video](https://cdn.discordapp.com/attachments/709432890458374204/1219546130115727390/image.png?ex=66309bd7&is=661e26d7&hm=c48cbefebdc49abcba54b0350bd200d4fae5accf0a629c695a429e82c0eac7f9&)](https://drive.google.com/file/d/1R7l4302yjyHZzcCP7Cm9vKr7sSnPDih_/view)


![](https://cdn.discordapp.com/attachments/1224691102284648448/1260213293596016681/point_cloud_animation.gif?ex=6695c0d6&is=66946f56&hm=8418246cc3e75192689b509672f2dea22b8bf7c9729f6ec4e562e8f1532d0c99&)

## Development environment:

### VS-Code:
The project is designed to be delevoped within vs-code IDE using remote container development.

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
> [!CAUTION]
> xhost + is not a save operation!
## Training:
### Train Semantic Kitti

Run a single training by:
```bash
appuser@a359158587ad:~/repos$ python train_semantic_Kitti.py --model_type resnet34 --learning_rate 0.001 --num_epochs 50 --batch_size 1 --num_workers 1 --rotate --flip --visualization
```

Run all trainings by:
```bash
appuser@a359158587ad:~/repos$ chmod +x run_training_kitti.sh
appuser@a359158587ad:~/repos$ ./run_training_kitti.sh
```

### Train Semantic THAB

Run the training by:
```bash
python src/train_semantic_THAB.py --model_type resnet34 --learning_rate 0.001 --num_epochs 50 --batch_size 8 --num_workers 16 --rotate --flip --visualization
```

## Model Zoo
We provide a large collection of pre-trained models with different backbones, number of parameters, and inference times.
You can choose the model suitable for your application.

### SemanticKitti
![image info](./Images/Inference_KITTI.png)

You can download pre-trained models from our model zoo:
| Backbone | Parameters | Inference Time¹ | mIoU² | Status 
|:--------:|:----------:|:---------------:|:----:|:------:|
| [resnet18](https://drive.google.com/drive/folders/1pPLbw5a5TwYnK77cjaEN5NBfhnHfZf6q?usp=sharing) |  18.5 M     |  9.8 ms  | 55.6%  | $${\color{green}Online}$$ 
| [resnet34](https://drive.google.com/drive/folders/16VT7LU-s9LloC3w-wi2rpBeztWs1Z2az?usp=sharing) |  28.3 M      |  13.6ms  | 57.3%  | $${\color{green}Online}$$ 
| [resnet50](https://drive.google.com/drive/folders/1bmO6shunZU20Rsr4_vZkjwXy9Av7Cmz3?usp=sharing) |  128.8 M      |  43.7 ms  | 60.07%  | $${\color{green}Online}$$ 
| [regnet_y_400mf](https://drive.google.com/drive/folders/178phFeiDOuvMP9ML3qrivjfacUzie-JG?usp=sharing) |  8.6 M      |  14.2 ms  | 55.0%  | $${\color{green}Online}$$ 
| [regnet_y_800mf](https://drive.google.com/drive/folders/1jaRdFrJR2vugUYf06tdT71WW0xr6oo74?usp=sharing) |  16.7 M      |  14.4 ms  | 55.64%  | $${\color{green}Online}$$ 
| [regnet_y_1_6gf](https://drive.google.com/drive/folders/1jHjeNciRzfXASVsqWPCXW3W-ZW6lc4a8?usp=sharing) |  22.25 M      |  21.7 ms  | 55.78%  | $${\color{green}Online}$$ 
| [regnet_y_3_2gf](https://drive.google.com/drive/folders/1f9UA4r6NWrIMmw0I9k0iNTVekjClYtmG?usp=sharing) |  52 M      |  25.1 ms  | 55.69%  | $${\color{green}Online}$$
| [shufflenet_v2_x0_5](https://drive.google.com/drive/folders/1nk4eHfZEgeP5NBjV65vxaJfQYd8HINOg?usp=sharing) |  4.3 M      |  10.24 ms  | 53.6%  | $${\color{green}Online}$$
| [shufflenet_v2_x1_0](https://drive.google.com/drive/folders/1OejQWT_PiGh-Y3RSVfCfOq2GKQUuhCr_?usp=sharing) |  13.2 M      |  15.1 ms  | 58.0%  | $${\color{green}Online}$$
| [shufflenet_v2_x1_5](https://drive.google.com/drive/folders/1VVg2ns76OCPIPb2_u-nxhk7m2L8hWtEq?usp=sharing) |  25.1 M      |  23.6 ms  | 59.38%  | $${\color{green}Online}$$



¹ Inference time measured as forward path time at a Nivida Geforce RTX 2070 TI with batchsize of one @ 128x2048 resolution. Only the forward path through the ML Model is measured!

² mIoU is measured in range view representation. NaNs (from non occuring classes in SemanticKitti Val) are treated as zeros.
  IoU results are not directly comparable to the SemanticKitti benchmark! 


## Inference:
You can explore /src/inference_ouster.py for an example how to use our method with a data stream from an Ouster OS2-128 sensor.
We provide a sample sensor recording.

<a name="license"></a>
## License:
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

### References
[1]   J. Behley et al., "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 9296-9306, doi: 10.1109/ICCV.2019.00939.


