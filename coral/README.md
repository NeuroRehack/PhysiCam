# PyCoral Movenet Video Example

This directory contains examples that show how to use the
[PyCoral API](https://coral.ai/docs/edgetpu/api-intro/) to perform
inference or on-device pose estimation using the pre-trained Movenet model.

## Run the example

1.  [Set up your Coral device](https://coral.ai/docs/setup/) (includes steps
to install this PyCoral library).

2.  Clone this repo onto the host device (onto your Coral board, or if using
a Coral accelerator, then onto the host system):

    ```
    git clone https://github.com/NeuroRehack/PhysiCam.git
    ```

3.  Download the required model and other files required to run each sample,
using the `install_requirements.sh` script.

    ```
    bash install_requirements.sh
    ```

4.  Connect USB Accelerator to the host device.

5.  Then run the example command shown at the top of the `.py` file to run
the code. 

<!--
Some examples also require
additional downloads, which are specified in the code comments at the top of the
file.
-->

## Movenet Pose Estimation

### Movenet Landmarks
```
0: 'nose'
1: 'left_eye'
2: 'right_eye'
3: 'left_ear'
4: 'right_ear'
5: 'left_shoulder'
6: 'right_shoulder'
7: 'left_elbow'
8: 'right_elbow'
9: 'left_wrist'
10: 'right_wrist'
11: 'left_hip'
12: 'right_hip'
13: 'left_knee'
14: 'right_knee'
15: 'left_ankle'
16: 'right_ankle'
```

### Example Output
```
[0.47367048 0.6101518  0.5017696 ]
[0.42148647 0.6221943  0.3733166 ]
[0.42951477 0.59409523 0.5017696 ]
[0.46162802 0.6302226  0.18063706]
[0.4375431  0.51381207 0.26493436]
[0.71853405 0.59409523 0.7345907 ]
[0.66635    0.38134488 0.5017696 ]
[0.9593835  0.61818016 0.5017696 ]
[0.96741176 0.30909008 0.3733166 ]
[0.32916087 0.6061377  0.18063706]
[0.32916087 0.5258545  0.18063706]
[1.0115675  0.6462792  0.11641055]
[1.0035392  0.45359972 0.11641055]
[0.7305765  0.59810936 0.04816988]
[0.75867563 0.5419112  0.01204247]
[0.36930242 0.5860669  0.18063706]
[0.28500512 0.5459253  0.18063706]
```

For more pre-compiled models, see [coral.ai/models](https://coral.ai/models/).

For more information about building models and running inference on the Edge
TPU, see the [Coral documentation](https://coral.ai/docs/).
