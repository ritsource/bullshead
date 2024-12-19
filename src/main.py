import argparse
import sys
import os
import numpy as np
from serve.app import create_app
from data.preprocessing import preprocess_data
from model_interfaces.LSTM import LSTM
from model_interfaces.HMM import HMM

def main():
    parser = argparse.ArgumentParser(description='ML Model CLI')
    parser.add_argument('--serve', action='store_true', help='Start the API server')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--lstm', action='store_true', help='Use LSTM model')
    parser.add_argument('--hmm', action='store_true', help='Use HMM model')
    parser.add_argument('--backtest', action='store_true', help='Run model backtest (HMM only)')
    
    args = parser.parse_args()

    if args.lstm:
        model = LSTM()
    elif args.hmm:
        model = HMM()
    else:
        print("Please specify model type using --lstm or --hmm")
        sys.exit(1)
    
    if args.backtest:
        if not args.hmm:
            print("Backtest is only available for HMM model")
            sys.exit(1)
        results = model.backtest(5)
        print(f"Backtest Results: {results}")
    elif args.serve:
        app = create_app()
        app.run(host="0.0.0.0", port=8000)
    elif args.preprocess:
        preprocess_data(data_file_path="scraper/binance_data_merged/klines/30m/merged.csv")
    elif args.train:
        if args.lstm:
            model.train(sequence_size=60, target_attr_idx=0)
        else:
            model.train()
    elif args.test:
        if args.lstm:
            model.predict(sequence_size=60, target_attr_idx=0)
        else:
            _, _, test_data = model.construct_data()
            model.predict(test_data)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
