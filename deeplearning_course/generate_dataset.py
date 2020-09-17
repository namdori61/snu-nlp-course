from typing import Dict, List
from collections import defaultdict
from random import randint
from tqdm import tqdm
import logging
import argparse
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='Generate dataset.')
parser.add_argument('-n', '--num_data', type=int, default=10,
                    help='The number of data to generate.')
parser.add_argument('-m', '--max_coordinate', type=int, default=20,
                    help='Maximum coordinate value of points.')
parser.add_argument('-o', '--output_path', type=str, default=dir_path + '/input.txt',
                    help='Output path to save input dataset.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = parser.parse_args()


class Gen2dData:
    def __init__(self,
                 num_data: int = 10,
                 max_coordinate: int = 20) -> None:
        self.num_data = num_data
        self.max_coordinate = max_coordinate

    def gen_triangles(self,
                      num_data: int = 10,
                      max_coordinate: int = 20) -> Dict[int, List[List[int]]]:
        t = defaultdict(list)
        for n in tqdm(range(num_data), desc='generating triangles'):
            t[n] = [[randint(0, max_coordinate), randint(0, max_coordinate)],
                    [randint(0, max_coordinate), randint(0, max_coordinate)],
                    [randint(0, max_coordinate), randint(0, max_coordinate)]]
        return t

    def gen_affine_transformation(self,
                                  max_coordinate: int = 20) -> Dict[str, List[List[int]]]:
        affine = defaultdict(list)
        affine['matrix'] = [[randint(0, max_coordinate), randint(0, max_coordinate)],
                            [randint(0, max_coordinate), randint(0, max_coordinate)]]
        affine['bias'] = [randint(0, max_coordinate), randint(0, max_coordinate)]
        return affine


def main():
    logger.info('You have set needed arguments like below:')
    logger.info(f'Generate {args.num_data} triangles.')
    logger.info(f'Maximum coordinate value is {args.max_coordinate}')
    logger.info(f'Dataset will be saved in {args.output_path}')

    generator = Gen2dData(num_data=args.num_data,
                          max_coordinate=args.max_coordinate)
    dataset = generator.gen_triangles(num_data=generator.num_data,
                                      max_coordinate=generator.max_coordinate)
    affine = generator.gen_affine_transformation(max_coordinate=generator.max_coordinate)
    input_dataset = (dataset, affine)

    with open(args.output_path, 'wb') as output_file:
        pickle.dump(obj=input_dataset, file=output_file)


if __name__ == '__main__':
    main()
