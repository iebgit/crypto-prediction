
# Bitcoin Price Prediction using LSTM

This project uses an LSTM (Long Short-Term Memory) neural network to predict future Bitcoin prices based on historical weekly closing prices.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/bitcoin-price-prediction.git
   cd bitcoin-price-prediction
   ```

2. **Create a virtual environment and activate it**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

## Training the Model

To train the LSTM model on the historical Bitcoin prices, run the `train_lstm.py` script:

```sh
python train_lstm.py
```

This will:
- Load the data from `bitcoin_weekly_closing_prices.csv`
- Normalize the data
- Create sequences of data for the LSTM
- Split the data into training and testing sets
- Build and train the LSTM model
- Save the trained model in `SavedModel` format as `bitcoin_lstm_model.keras`

## Predicting Future Prices

To predict future Bitcoin prices, run the `predict_future_price.py` script:

```sh
python predict_price.py
```

This will:
- Load the trained model
- Load and normalize the data
- Prepare the input data for prediction
- Iteratively predict future prices
- Calculate and print the predicted price on a specified future date

## Example

To predict the Bitcoin price on a specific date, ensure the `predict_price.py` script has the correct target date set and run it:

```sh
python predict_price.py
```

The output will display the predicted price on the specified future date.

## Contributing

Feel free to fork this repository, create a branch, and submit a pull request if you'd like to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses TensorFlow and Keras for building and training the LSTM model. Special thanks to the open-source community for providing these powerful tools.

