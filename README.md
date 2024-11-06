
# Movie Recommendation System with Graph Neural Networks (GNN)

This project implements a movie recommendation system using a Graph Neural Network (GNN) model. It leverages the relationships between users and movies to predict user-movie ratings based on graph-based representation and processing.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Architecture](#model-architecture)
4. [Hyperparameters](#hyperparameters)
5. [Training and Evaluation](#training-and-evaluation)
6. [Sample Training Output](#sample-training-output)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hanish0/Movie-Recommendation-System-with-Graph-Neural-Networks-GNN
   cd Movie-Recommendation-System-with-Graph-Neural-Networks-GNN
   ```

2. **Set up the virtual environment**:
   ```bash
   python3 -m venv recommender_env
   source recommender_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MovieLens Dataset**:
   - Run the provided download script or download the dataset manually from the [MovieLens website](https://grouplens.org/datasets/movielens/).
   - Place the dataset in the appropriate directory as specified in the project configuration.

## Usage

To train the model and evaluate it, follow these steps:

1. **Activate the virtual environment**:
   ```bash
   source recommender_env/bin/activate
   ```

2. **Run the main script**:
   ```bash
   python3 main.py
   ```

This will train the model on the MovieLens dataset and output the evaluation results.

## Model Architecture

The recommendation system uses a Graph Neural Network (GNN) model with the following key components:

- **Input Layer**: Encodes genre features for movies and random features for users.
- **Graph Convolutional Layers (GCNConv)**: Captures relationships between user and movie nodes to learn user preferences and item characteristics.
- **Output Layer**: Provides a prediction for user-movie ratings.

## Hyperparameters

The model's key hyperparameters are set as follows:

- **Hidden Layer Size**: 64
- **Learning Rate**: 0.01
- **Training Epochs**: 100

## Training and Evaluation

The model is trained on user-movie interactions, with actual ratings as edge labels. The primary evaluation metrics are:

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**

## Sample Training Output

The training output below demonstrates the model’s learning progress over 100 epochs. Key metrics, such as loss and validation scores, provide insights into the model’s performance.

```
Training Progress:
------------------

Epoch 0, Loss: 11.2374
Epoch 10, Loss: 1.8917
Epoch 20, Loss: 1.3708
Epoch 30, Loss: 1.2776
Epoch 40, Loss: 1.1282
Epoch 50, Loss: 1.0751
Epoch 60, Loss: 1.0467
Epoch 70, Loss: 1.0275
Epoch 80, Loss: 1.0138
Epoch 90, Loss: 1.0035

Final Validation Results:
-------------------------
Mean Squared Error (MSE): 1.0164
Root Mean Squared Error (RMSE): 1.0082
R-squared (R²): 0.0761
```

In this output:
- **Initial Loss (Epoch 0)**: The loss starts high, indicating that the model is just beginning to learn and has not yet adjusted weights to minimize error.
- **Improvement over Epochs**: By epoch 10, there is a significant drop in loss, showing that the model is learning to fit the data. This trend continues with gradual improvements.
- **Final Validation Metrics**: After 100 epochs, the model reaches an MSE of 1.0164 and an R² of 0.0761. These metrics suggest moderate predictive performance, with room for further tuning and feature enhancement.

## Results

The model achieved the following results on the validation set:

- **Mean Squared Error (MSE)**: 1.0164
- **Root Mean Squared Error (RMSE)**: 1.0082
- **R-squared (R²)**: 0.0761

These results indicate that while the model captures some patterns in user preferences, additional improvements may be necessary for higher predictive accuracy.

## Future Work

Potential improvements to the model include:

- **Additional User and Movie Features**: Incorporating demographic data, more detailed movie descriptions, or other side information.
- **Advanced Architectures**: Experimenting with deeper GNN layers, attention mechanisms, or hybrid models combining GNNs with other architectures.
- **Hyperparameter Tuning**: Optimizing learning rates, hidden layer sizes, and other parameters for better performance.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

Feel free to open an issue if you encounter any problems or have questions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

