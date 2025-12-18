# RheoHierarch

## Installation
### Ml-agents
Please follow the official tutorial to install mlagent==0.28.0:

[ML-Agents Official Installation Guide](https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs/Installation.md)

### Fluidlab
Please follow the official tutorial to install fluidlab:
[Fluidlab](https://github.com/zhouxian/FluidLab)


## Software
* Ubuntu 20.04/22.04
* Unity 2021.3.11f1c2
* RealSense SDK v2.53.1


## Guide
- Pre-training
  - Prepare the Unity ML-Agents initial environment by following the [official tutorial](https://github.com/Unity-Technologies/ml-agents), with additional perception dependencies from [mbaske/grid-sensor](https://github.com/mbaske/grid-sensor) for specific tasks (e.g., pour, gather).
  - Get pre-trained model

    ```bash
    #demo
    conda activate mlagents
    cd ./ml-agents/ml-agents/mlagents/trainers
    python learn.py ~/RheoAgent/external/ml-agents/config/poca/Fluidlab.yaml --run-id=pour --num--envs=10 --base-port=5004 --force --env=/your/env/name --no-graphics
    ```
- Fine-tuning
  - Get fine-tuned model
    ```bash
    #demo
    conda activate fluidlab
    cd ./RheoAdapt
    python run.py --cfg_file configs/shac/flow.yaml --rl shac --exp_name=water --perc_type sensor --pre_train_model init_policy.pt --horizon 500 --material WATER
    ```

    - **pre_train_model**: Use the initial model obtained in [Pre-training](#Pre-training) as the pre-trained model
    - **rl**: Use SHAC reinforcement learning algorithm for fine-tuning

- Evaluation in the real-world
  - [External Hand-Eye Calibration](https://github.com/pal-robotics/aruco_ros)
  - [Controller Setup](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)(For Univeral Robots)
  - [SAM2 Visual Perception](https://github.com/facebookresearch/sam2)
  - End-to-End Control Publishing

