import logging
import argparse
import pickle
import os
from pprint import pprint

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='Show results.')
parser.add_argument('-i', '--input_path', type=str, default=dir_path + '/input.txt',
                    help='Input path to load input dataset.')
parser.add_argument('-o', '--output_path', type=str, default=dir_path + '/output.txt',
                    help='Output path to load output dataset.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = parser.parse_args()


def main():
    logger.info(f'Load input data from {args.input_path}')
    with open(args.input_path, 'rb') as input_file:
        input_dataset = pickle.load(input_file)

    logger.info(f'Load output data from {args.output_path}')
    with open(args.output_path, 'rb') as output_file:
        output_dataset = pickle.load(output_file)

    print('Input dataset:')
    pprint(input_dataset)
    print('-' * 100)
    print('Output dataset:')
    pprint(output_dataset)


if __name__ == '__main__':
    main()