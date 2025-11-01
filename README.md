# SimSiam: Self-Supervised and Contrastive Representation Learning

This project, developed for an M.S. Deep Learning course, implements and compares three representation learning strategies on the CIFAR-10 dataset: a supervised baseline, a multi-task pretext (self-supervised) model, and a SimSiam (contrastive) model. The primary goal is to demonstrate the effectiveness of self-supervised and contrastive methods in a low-data regime, where only a small subset of labeled data (5,000 images) is available for (fine-)tuning.

## Features

* **Supervised Baseline:** A standard CNN trained from scratch on only 5,000 labeled images.
* **Self-Supervised (Pretext):** A model pre-trained on all 50,000 images to predict augmentations (rotation, shear, color) before being fine-tuned on the 5k labeled set.
* **Contrastive (SimSiam):** A Siamese network pre-trained on all 50,000 images using a stop-gradient, negative-pair-free contrastive loss (SimSiam).
* **Comparative Analysis:** Includes scripts and a notebook to train all models and generate comparative plots of their performance, demonstrating the power of SSL/Contrastive learning.
* **Modular & Robust:** The project is structured with modular code, command-line arguments (`argparse`), and comprehensive logging (to console and files).

## Core Concepts & Techniques

* Supervised Learning (Baseline)
* Self-Supervised Learning (SSL) via Pretext Tasks
* Contrastive Learning (SimSiam)
* Representation Learning
* Transfer Learning (Fine-tuning vs. Linear Probing)
* Convolutional Neural Networks (CNNs)
* PyTorch, `argparse`, Logging

---

## How It Works

This project compares three distinct methods to train a model for image classification, highlighting the challenge of low-labeled data.

### 1. Overview & Project Goal

The core problem is that modern deep learning models require vast amounts of *labeled* data, which is expensive to acquire. Supervised learning fails in low-data regimes. Self-Supervised Learning (SSL) and Contrastive Learning aim to solve this by first learning rich feature representations from a large pool of *unlabeled* data. This pre-trained "backbone" can then be quickly adapted to a downstream task (like classification) using only a few labels.

This project implements and compares:
1.  **Baseline:** Training from scratch on 5k labels.
2.  **SSL (Pretext):** Pre-training on 50k unlabeled images (pretext task) + fine-tuning on 5k labels.
3.  **SimSiam (Contrastive):** Pre-training on 50k unlabeled images (contrastive task) + linear probing on 5k labels.

### 2. Algorithms & Models

All models share an identical backbone architecture (defined in `src/models.py`) for fair comparison.

#### 1. Baseline Model (`scripts/train_baseline.py`)

* **Architecture:** A simple CNN (`src/models.py:Network`) is trained from scratch.
* **Training:** The model is trained *only* on the 5,000-image labeled subset of CIFAR-10.
* **Expected Outcome:** This model is expected to overfit severely and achieve a low test accuracy, setting the "lower bound" for performance.

#### 2. Self-Supervised Pretext Model (`scripts/train_ssl_pretext.py`)

* **Architecture:** A multi-head CNN (`src/models.py:Supervised_Learning`) with a shared backbone and four separate output heads:
    1.  Rotation Head (4 classes: 0°, 90°, 180°, 270°)
    2.  Shear Head (3 classes: 0.0, 0.2, 0.4)
    3.  Color Head (3 classes: original, grayscale, inverted)
    4.  Classification Head (10 classes)
* **Training (Phase 1 - Pre-training):** The model is trained on all 50,000 CIFAR-10 images. For each image, a random set of augmentations (rotation, shear, color) is applied. The model is trained to predict *which* augmentations were applied, using a combined loss from the three pretext heads. This forces the backbone to learn meaningful features (e.g., "what does a 'normal' orientation look like?").
* **Training (Phase 2 - Fine-tuning):** The pre-trained model is then fine-tuned on the 5,000-image labeled set, using only the classification head and loss.

#### 3. Contrastive SimSiam Model (`scripts/train_siam.py`)

* **Architecture:** The SimSiam model (`src/models.py:SimSiam`) consists of a Siamese network architecture:
    * **Encoder ($f$):** A backbone (the same as baseline) followed by a 3-layer MLP "projector." This maps an image $x$ to a representation $z = f(x)$.
    * **Predictor ($h$):** A 2-layer MLP that transforms the representation from one branch: $p = h(z)$.
* **Training (Phase 1 - Pre-training):**
    1.  Two "views" ($x_1, x_2$) are created from the same image using strong augmentations.
    2.  Both views are passed through the encoder: $z_1 = f(x_1)$ and $z_2 = f(x_2)$.
    3.  One branch's representation is passed through the predictor: $p_1 = h(z_1)$ and $p_2 = h(z_2)$.
    4.  The model is trained to make the prediction from one branch ($p_1$) match the representation from the *other* branch ($z_2$), and vice-versa.
* **Loss Function:** The model minimizes the negative cosine similarity.

  $$L = \frac{1}{2} D(p_1, z_2) + \frac{1}{2} D(p_2, z_1)$$

  Where $D(p, z) = - \frac{p}{\|p\|_2} \cdot \frac{z}{\|z\|_2}$
* **Stop-Gradient:** This is the *most important* concept in SimSiam. By stopping the gradient from $z$ on one side, we prevent the model from collapsing to a trivial solution (e.g., outputting the same constant for all inputs). It forces the predictor $h$ to learn to *predict* the stable representation $z$ from the other branch.
* **Training (Phase 2 - Linear Probing):** The backbone is *frozen*. A new classification head (`src/models.py:ClassificationModel`) is attached and trained *only* on the 5,000-image labeled set.

### 3. Analysis of Results

* **Baseline:** As expected, the baseline model overfits significantly. Training on only 10% of the data causes the training accuracy to skyrocket while the test accuracy plateaus at a low level (we see **~63-65%**). This demonstrates the limitation of supervised learning in low-data regimes.
* **SSL (Pretext):** The pretext-task model performs slightly better, reaching **~67-68%** test accuracy. The pre-training on augmentations forces the backbone to learn more robust features (like orientation, basic shapes), which provides a better starting point for fine-tuning. However, the model still overfits, as the entire network is fine-tuned on the small 5k dataset.
* **SimSiam (Linear Probe):** The SimSiam model shows the most stable behavior. The linear probe (training *only* the classifier on the *frozen* backbone) is highly resistant to overfitting. While its peak accuracy (**~65-66%**) is slightly lower than the fully fine-tuned SSL model, it's more robust and achieves this by training *far fewer* parameters. This highlights the key benefit: the backbone, trained on all 50k unlabeled images, has learned a powerful, general-purpose representation of the data. This is confirmed by the UMAP visualization in the notebook, which shows clear clustering of the data by class, *before* the classifier was even trained.

---

## Project Structure

```
pytorch-simsiam-contrastive-ssl/
├── .gitignore                  # Ignores data, logs, outputs, and venv
├── LICENSE                     # MIT License
├── README.md                   # You are here
├── requirements.txt            # Python dependencies
├── logs/
│   └── .gitkeep                # Logs from training runs will be saved here
├── outputs/
│   └── .gitkeep                # Saved models (.pth), plots (.png), and histories (.json)
├── notebooks/
│   └── run_experiments.ipynb   # Main notebook to run all scripts and analyze results
└── src/
│   ├── __init__.py
│   ├── data_loader.py          # Contains all Dataset and DataLoader logic
│   ├── models.py               # Contains all nn.Module definitions
│   └── utils.py                # Contains helper functions (logging, plotting)
└── scripts/
    ├── __init__.py
    ├── train_baseline.py       # Script to train the supervised baseline
    ├── train_ssl_pretext.py    # Script for pretext SSL pre-training and fine-tuning
    └── train_siam.py           # Script for SimSiam pre-training and linear probing
````

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-simsiam-contrastive-ssl.git
    cd pytorch-simsiam-contrastive-ssl
    ```

2.  **Setup Environment and Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the Experiments:**
    You can run each experiment individually using the scripts:
    ```bash
    # Run the baseline model (50 epochs)
    python scripts/train_baseline.py --epochs 50

    # Run the SSL Pretext model (30 pretext epochs, 30 finetune epochs)
    python scripts/train_ssl_pretext.py --pretext_epochs 30 --finetune_epochs 30

    # Run the SimSiam model (30 pretrain epochs, 50 probe epochs)
    python scripts/train_siam.py --pretrain_epochs 30 --probe_epochs 50
    ```

4.  **Analyze Results (Recommended):**
    For a guided walkthrough that runs all scripts and generates the final comparison plots and UMAP visualizations, use the main notebook:
    ```bash
    jupyter notebook notebooks/run_experiments.ipynb
    ```
    *All generated models, logs, and plots will be saved in the `outputs/` and `logs/` directories.*

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
