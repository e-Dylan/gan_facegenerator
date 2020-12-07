# gan_facegenerator
Face generation GAN algorithm which then gets converted into cartoon characters and animated using AI.

## Training progress visualization.
![App Demo](demo/training_visual.gif)

## Training Process
15-minute training loss graph for 64 x 64 groundtruth images.
![Train Loss Graph](train-loss-graph.png)

Training is done by feeding 64-sized batches of face images using the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. Images were scaled to 128x128 and the network architecture was designed accordingly. The network learns to extract features from human faces and replicate them artificially by gradient descent.

Training was done on a single GPU for roughly 3 hours. The final product generates believable human faces at 128x128 resolution. These are visible at [/demo](https://github.com/e-Dylan/gan_facegenerator/tree/master/demo)
