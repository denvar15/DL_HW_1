# DL_HW_1
Skoltech DL course 2025

**Suchkov Denis, Boldyrev Nikita** team

# My Experiments

## Models

During the two weeks given to complete the homework, I tried the following methods to solve the problem:

1.   EfficientNetV2 from pytorch models (see the code in Kaggle colab at the links below + efficientnet_v2_m.ipynb) - best result 0.83,
2.   SWIN_v2 transformer from pytorch models (see the code in Kaggle colab at the links below + swin_v2_t 30 epochs.ipynb) - best result 0.85,
3.   ViT vit_base_patch16_224 from timm (see the code in Google colab at the links below + dl_hw_vit.ipynb) - best result 0.86,
4.   NFNet nfnet_l0.ra2_in1k from timm (see the code in Kaggle colab at the links below + nfnet 90 epochs.ipynb) - best result 0.88,
5.   ConvNeXt zoobot-encoder-convnext_tiny from zoobot (see the code in Google colab at the links below + DL_HW_1_Suchkov_Denis_Boldyrev_Nikita.ipynb) - best result 0.89.

## Methods

For these networks, I used the following methods to combat overfitting and improve finetuning:



1.   A classifier head with not just a linear output for 10 classes, but with nonlinearity and Dropout to improve classification (see the code in Google and Kaggle colab at the links below),
2.   Used different Dropout values ​​in the improved head and the last layers of the networks (see the code in Kaggle colab at the links below),
3.   Tried to tune not all layers of the network, but only the last NUM_UNFROZEN_LAYERS (for example, 20) layers to avoid complex optimization (see the code in Google and Kaggle colab at the links below),
4.   Used various augmentations, including normalization on the calculated values ​​of the mean and std by the channels of the training part of the dataset, rotations, random crop (see the code in Kaggle colab at the links below),
5.   Used training with Focal Loss instead of the usual one CrossEntropyLoss to solve the problem of poor classification of the zeroth class of the dataset by most models (see the code in Google colab at the links below),
6.   Used a weighted CrossEntropyLoss by the number of elements of the classes in the dataset to improve the classification of classes with low support (see code in Google and Kaggle colab at the links below),
7.   Used TTA to solve the problem with model overfitting (see code in Kaggle colab at the links below),
8.   Trained models of different models (e.g. swin_t, swin_s, swin_b sequentially) to find the relevant model complexity for the problem (see code in Kaggle colab at the links below),
9.   Used label_smoothing in CrossEntropyLoss to solve the classification problem on poorly predictable classes (see code in Google and Kaggle colab at the links below),
10.  Used CosineAnnealingLR to improve the convergence of the optimizer and avoid hitting a local minimum (see code in Google and Kaggle colab at the links below).

## Code snippets for methods



1. 
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10)
        )
2.  
        model.head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(256, 10)
        )
3. 
        for param in model.features.parameters():
          param.requires_grad = False

        for epoch in range(EPOCHS_FREEZE):
          model.train()
          total_loss = 0
          for imgs, labels in train_dataloader:
              imgs, labels = imgs.to(device), labels.to(device)
              optimizer.zero_grad()
              output = model(imgs)
              loss = cost(output, labels)
              loss.backward()
              optimizer.step()
              total_loss += loss.item()

          print(f"[Epoch {epoch+1}/{EPOCHS_FREEZE}] Freeze Phase Loss: {total_loss/len(train_dataloader):.4f}")

        params_to_unfreeze = []
        for name, param in reversed(list(model.features.named_parameters())):
          if NUM_UNFROZEN_LAYERS <= 0:
              break
          param.requires_grad = True
          params_to_unfreeze.append(name)
          NUM_UNFROZEN_LAYERS -= 1

        print("Unfrozen layers:")
        print(params_to_unfreeze)
4.  
        def get_transforms(train=True):
            if train:
                transform = A.Compose([
                    A.RandomResizedCrop(size=(CFG["image_size"], CFG["image_size"]), scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(p=0.5),
                    A.Normalize(mean=[0.16, 0.16, 0.16], std=[0.12, 0.1, 0.1]),
                    ToTensorV2(),
                ])
            else:
                transform = A.Compose([
                    A.Resize(CFG["image_size"], CFG["image_size"]),
                    A.CenterCrop(CFG["image_size"], CFG["image_size"]),
                    A.Normalize(mean=[0.16, 0.16, 0.16], std=[0.12, 0.1, 0.1]),
                    ToTensorV2(),
                ])
            return transform
5.  
        class GalaxyFocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, n_classes=10, device='cuda'):
                """
                Focal Loss with:
                - Class-balanced alpha (optional)
                - Label smoothing
                - Focus parameter gamma

                Args:
                    alpha (Tensor, optional): Class weights. Tensor of size [n_classes]
                    gamma (float): Focusing parameter (0 = CE, >1 reduces easy example impact)
                    smoothing (float): Label smoothing epsilon (0-0.2 recommended)
                    n_classes (int): Number of classes (10 for Galaxy10)
                    device: Device to use
                """
                super(GalaxyFocalLoss, self).__init__()
                self.gamma = gamma
                self.smoothing = smoothing
                self.n_classes = n_classes

                if alpha is None:
                    self.alpha = None
                else:
                    self.alpha = alpha.to(device)

                self.device = device

            def forward(self, inputs, targets):
                # Convert targets to one-hot & apply smoothing
                targets_onehot = F.one_hot(targets, num_classes=self.n_classes).float()
                targets_onehot = (1.0 - self.smoothing) * targets_onehot + self.smoothing / self.n_classes

                # Compute softmax probabilities
                log_prob = F.log_softmax(inputs, dim=1)
                prob = torch.exp(log_prob)

                # Focal Loss calculation
                focal_weight = (1 - prob) ** self.gamma
                loss = -targets_onehot * focal_weight * log_prob

                # Class balancing
                if self.alpha is not None:
                    alpha_weight = self.alpha[targets].view(-1, 1)
                    loss = alpha_weight * loss

                return loss.mean()   
6.  
          class_counts = np.bincount(galaxy_dataset['train']['label'])  # e.g., [109, 185, ..., 178] for Galaxy10
          num_samples = sum(class_counts)
          class_weights = num_samples / (len(class_counts) * class_counts)  # Inverse frequency weighting
          weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
          criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
7. 
        tta_transforms = [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
                A.NoOp(p=1.0),  # Identity transform
            ]
8.    Just different runs in Kaggle colab version history
9. 
          criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
10.
          scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE)


## Links



*   Main Google colab notebook: https://colab.research.google.com/drive/1tHM3tqRIyt-Wl0UQlTqUja6g0EC3JKAD?usp=sharing
*   Kaggle colab notebook: https://www.kaggle.com/code/denvar15/dl-hw-1
*   Additional Google colab notebook (some expriments with ViT): https://colab.research.google.com/drive/1qpxi7MBe4Mhfgf0AYOP_xdSiXCO8rabo?usp=sharing


