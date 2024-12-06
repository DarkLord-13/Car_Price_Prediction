# Car Price Prediction

This project aims to predict the selling price of cars using machine learning techniques. The dataset contains information about various cars, including their features and selling prices.

## Project Overview

The project follows these main steps:
1. **Data Preparation**:
    - Load the dataset.
    - Perform exploratory data analysis.
    - Handle missing values and drop irrelevant columns.
    - Encode categorical variables.
2. **Model Training**:
    - Split the data into training and testing sets.
    - Train and evaluate two models: Linear Regression and Lasso Regression.
3. **Model Evaluation**:
    - Compare the performance of the models using metrics such as R-squared score.
    - Visualize the actual vs predicted prices.

## Dependencies

The project requires the following dependencies:
- Python 3.x
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/DarkLord-13/Machine-Learning-01.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Machine-Learning-01
    ```

3. Install the required packages:
    ```sh
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```

4. Open the Jupyter Notebook `CarPricePrediction.ipynb` and run the cells to execute the project steps:
    ```sh
    jupyter notebook CarPricePrediction.ipynb
    ```

## Usage

1. **Data Preparation**:
    - Load the dataset and perform exploratory data analysis.
    - Handle missing values and drop the `Car_Name` column.
    - Encode the categorical variables such as `Fuel_Type`, `Seller_Type`, and `Transmission`.

2. **Model Training**:
    - Split the data into training and testing sets using `train_test_split`.
    - Train a Linear Regression model and evaluate its performance.
    - Train a Lasso Regression model and evaluate its performance.

3. **Model Evaluation**:
    - Compare the R-squared scores of the Linear Regression and Lasso Regression models.
    - Visualize the actual vs predicted prices using scatter plots.

## Results

The trained models will predict the selling prices of cars based on their features. The performance of the models can be evaluated using the R-squared score and visualizing the actual vs predicted prices.

## License

This project is licensed under the MIT License.
