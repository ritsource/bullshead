import argparse
import sys
from serve.app import create_app

def main():
    parser = argparse.ArgumentParser(description='ML Model CLI')
    parser.add_argument('--serve', action='store_true', help='Start the API server')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    
    args = parser.parse_args()

    if args.serve:
        app = create_app()
        app.run(host="0.0.0.0", port=8000)
    elif args.train:
        print("train")
    elif args.test:
        print("test")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
