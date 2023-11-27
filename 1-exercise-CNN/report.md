# CNN

## Hyperparameters

- **Learning rate (LR)** : Determines the step size at which the model's weights are
  updated during the optimization process. The learning rate influences the
  trade-off between convergence speed and convergence quality. A smaller
  learning rate will take more time to converge but will be more precise, while
  a larger learning rate will converge faster but with a lower precision.

- **Momentum (MOMENTUM)** : Enhances the optimization process by accelerating
  convergence, especially in the presence of noisy or sparse gradients. Momentum
  is used to smooth out the updates to the model's weights and helps the
  optimizer overcome local minima and saddle points more effectively. Higher
  momentum value allows the optimizer to maintain a stronger "memory" of past
  gradients, which can help the optimization process overcome obstacles and
  converge faster. However, if the momentum is set too high, it may cause the
  optimization process to overshoot the optimal solution or lead to
  oscillations.

- **Weight decay (WEIGHT_DECAY)** : Often referred to as L2 regularization, is a
  technique used in machine learning and deep learning to prevent overfitting by
  adding a regularization term to the loss function. This regularization term
  encourages the model's weights to be smaller, thereby reducing the complexity
  of the model and making it less prone to fitting noise in the training data.

- **Number of epochs (NUM_EPOCHS)** : Total number of training epochs (iterations over dataset).

- **Step size (STEP_SIZE)** : How many epochs before decreasing learning rate by gamma.

- **Gamma (GAMMA)** : Multiplicative factor for learning rate step-down.

<br />

# Training from Scratch

Different models referes to different set of hyperparameters.

## Results

| Model   | LR   | MOMENTUM | WEIGHT_DECAY | NUM_EPOCHS | STEP_SIZE | GAMMA | TRAINING LOSS AT LAST STEP | VALIDATION ACC      | TESTING ACC         |
| ------- | ---- | -------- | ------------ | ---------- | --------- | ----- | -------------------------- | ------------------- | ------------------- |
| Base    | 1e-3 | 0.9      | 5e-5         | 30         | 20        | 0.1   | 4.254359245300293          | 0.08575380359612725 | 0.09194607673695127 |
| Model 1 | 1e-1 | 0.9      | 5e-5         | 30         | 10        | 0.1   | 3.464630365371704          | 0.24827109266943292 | 0.24541997926028344 |
| Model 2 | 1e-1 | 0.9      | 5e-5         | 40         | 10        | 0.01  | 3.1392180919647217         | 0.2773167358229599  | 0.2744555824403733  |
| Model 3 | 1e-2 | 0.9      | 5e-5         | 30         | 15        | 0.01  | 2.808311700820923          | 0.29322268326417705 | 0.29692360871068096 |

**Comments:**

- Base learning rate is too small, and combined combinet with a high gamma and
  high step size the model is not able to converge effectively in 30 epochs.
- Model 3 is the best model. A medium LR with lower step size and gamma seems to
  allow the model to converge better in the 30 epochs (3.23x accuracy from base model).

<br />

# Transfer Learning

AlexNet pretrained on Imagenet Dataset and finetuned on Caltech, starting from Base and Model 3.

Code changes from Base:

- `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- `net = alexnet(weights="DEFAULT")`

## Results

| Model   | LR   | MOMENTUM | WEIGHT_DECAY | NUM_EPOCHS | STEP_SIZE | GAMMA | TRAINING LOSS AT LAST STEP | VALIDATION ACC     | TESTING ACC        |
| ------- | ---- | -------- | ------------ | ---------- | --------- | ----- | -------------------------- | ------------------ | ------------------ |
| Base    | 1e-3 | 0.9      | 5e-5         | 30         | 20        | 0.1   | 0.04706478491425514        | 0.8333333333333334 | 0.8603525751814726 |
| Model 3 | 1e-2 | 0.9      | 5e-5         | 30         | 15        | 0.01  | 0.001122465473599732       | 0.8533886583679114 | 0.8627722087798133 |

**Comments:**

- Even though Base and Model 3 from scratch have different results, the transfer
  learning models benefit from the shared knowledge encoded in the lower layers
  of the pretrained model. This provides a strong starting point, allowing both
  transfer learning models to converge to similar results.

<br />

# Freezing Layers

AlexNet pretrained on Imagenet Dataset and finetuned on Caltech, starting from Model 3.

Code changes from Pretrained Model 3:

- Only Fully Connected:
  ```
  parameters_to_optimize=[]
  for name, param in net.named_parameters():
      if "classifier" in name:
          parameters_to_optimize.append(param) # optimize fully connected
      else:
          param.requires_grad = False # freeze not fully connnected
  ```
- Only Convolutional:
  ```
  parameters_to_optimize=[]
  for name, param in net.named_parameters():
      if "classifier" not in name:
          parameters_to_optimize.append(param) # optimize conv
          param.requires_grad = True
      else:
          param.requires_grad = False # freeze fully connnected
  ```

## Results

| Model                | LR   | MOMENTUM | WEIGHT_DECAY | NUM_EPOCHS | STEP_SIZE | GAMMA | TRAINING LOSS AT LAST STEP | VALIDATION ACC     | TESTING ACC        |
| -------------------- | ---- | -------- | ------------ | ---------- | --------- | ----- | -------------------------- | ------------------ | ------------------ |
| M3 - Fully Connected | 1e-2 | 0.9      | 5e-5         | 30         | 15        | 0.01  | 0.8627722087798133         | 0.8492392807745505 | 0.867611475976495  |
| M3 - Convolutional   | 1e-2 | 0.9      | 5e-5         | 30         | 15        | 0.01  | 1.971238613128662          | 0.5532503457814661 | 0.5620463187003111 |

**Comments:**

- Task Relevance: The fully connected layers are typically responsible for making
  task-specific predictions. In the case of the Caltech dataset, these top
  layers need to adapt to the unique characteristics and features of the
  dataset. By fine-tuning these layers, the model can learn to map the extracted
  features from the convolutional layers to the specific classes in the Caltech
  dataset.

- Layer Specificity: The convolutional layers of a pretrained model capture
  general image features, such as edges, textures, and basic shapes, which are
  often transferable across different computer vision tasks. These lower layers
  have already learned a lot of knowledge from the original ImageNet dataset. As
  a result, fine-tuning them may not be as critical for the specific task.

<br />

# Beyond AlexNet

ResNet pretrained on Imagenet Dataset and finetuned on Caltech, starting from Model 3.

Code changes from Model 3:

- `transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])`
- `net = resnet50(weights="DEFAULT")`
- `net.fc = nn.Linear(2048,NUM_CLASSES)`
- used Mixed Precision Training (AMP) to speed up training

## Results

| Model   | LR   | MOMENTUM | WEIGHT_DECAY | NUM_EPOCHS | STEP_SIZE | GAMMA | TRAINING LOSS AT LAST STEP | VALIDATION ACC     | TESTING ACC        |
| ------- | ---- | -------- | ------------ | ---------- | --------- | ----- | -------------------------- | ------------------ | ------------------ |
| Model 3 | 1e-2 | 0.9      | 5e-5         | 30         | 15        | 0.01  | 0.0003131479024887085      | 0.9156293222683264 | 0.9357068786726581 |

**Comments:**

- Mixed Precision Training (AMP) is a technique that can help speed up training
  while maintaining or even improving the model's accuracy. It leverages
  lower-precision data types for certain operations, which can reduce memory
  usage and computational requirements during training.

- ResNet-50 is a much deeper neural network compared to AlexNet. It has 50
  layers (hence the name) as opposed to AlexNet's 8 layers. Deeper networks can
  capture more complex features and relationships within the data.

- ResNet was a groundbreaking architecture that significantly advanced the field
  of deep learning. While AlexNet was a pioneering architecture, the subsequent
  advancements in neural network design, as exemplified by ResNet, have made it
  more effective for a broad spectrum of applications in the field of computer
  vision.
