from typing import Dict, List, Any
from collections import defaultdict
import pickle
from tqdm import tqdm
import logging
import argparse
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='Generate dataset.')
parser.add_argument('-i', '--input_path', type=str, default=dir_path + '/input.txt',
                    help='Input path to load dataset')
parser.add_argument('-o', '--output_path', type=str, default=dir_path + '/output.txt',
                    help='Output path to save results')
parser.add_argument('-p1', '--plot1_output_path', type=str, default=dir_path + '/first.png',
                    help='Output path to save first plot')
parser.add_argument('-p2', '--plot2_output_path', type=str, default=dir_path + '/second.png',
                    help='Output path to save second plot')
parser.add_argument('-nb', '--num_bins', type=int, default=5,
                    help='The number of bins to frequency table')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = parser.parse_args()


class Operator2dTriangle:
    def __init__(self,
                 input_path: str = None,
                 output_path: str = None,
                 plot1_output_path: str = None,
                 plot2_output_path: str = None) -> None:
        logger.info(f'Load dataset from {input_path}')
        with open(input_path, 'rb') as input_file:
            self.dataset = pickle.load(input_file)
        self.output_path = output_path
        self.plot1_output_path = plot1_output_path
        self.plot2_output_path = plot2_output_path

    # Methods related with problem 1
    def triangle_area(self,
                      tri: List = None) -> float:
        x1, y1, x2, y2, x3, y3 = tri[0][0], tri[0][1], tri[1][0], tri[1][1], tri[2][0], tri[2][1]
        return abs(0.5 * (((x2 - x1) * (y3 - y1)) - ((x3 - x1) * (y2 - y1))))

    def get_areas(self,
                  dataset: Dict = None) -> Dict[Any, float]:
        areas = defaultdict(float)
        logger.info(f'Get area of {len(dataset)} triangles')
        for n, t in tqdm(dataset.items(), desc='get areas'):
            areas[n] = self.triangle_area(t)
        return areas

    def get_areas_mean(self,
                       dataset: Dict = None) -> float:
        logger.info(f'Get mean area of {len(dataset)} triangles')
        _sum = 0
        _n = 0
        for v in dataset.values():
            _sum += v
            _n += 1
        return _sum / _n

    def get_areas_std(self,
                      dataset: Dict = None,
                      mean: float = None) -> float:
        logger.info(f'Get std area of {len(dataset)} triangles')
        _sum = 0
        _n = 0
        for v in dataset.values():
            _sum += (v - mean) ** 2
            _n += 1
        return sqrt(_sum / (_n - 1))

    def get_histogram(self,
                      dataset: Dict = None,
                      n_bins: int = None) -> List[Any]:
        logger.info(f'Get frequency table from dataset with {n_bins} bins')
        data_list = list(dataset.values())
        bin_size = (max(data_list) - min(data_list)) // n_bins + 1
        bins = [b * bin_size for b in range(n_bins + 1)]
        histogram = defaultdict(int)
        for d in data_list:
            for b in range(len(bins)):
                if bins[b] <= d < bins[b + 1]:
                    histogram[(bins[b], bins[b + 1])] += 1
        return sorted(histogram.items())

    def plot_histogram(self,
                       histogram: List = None,
                       output_path: str = 'first.png') -> None:
        logger.info(f'Plot frequency table to {output_path}')
        x = [d[0][0] for d in histogram]
        height = [d[1] for d in histogram]
        plt.figure()
        plt.xticks(x, [str(t[0]) for t in histogram])
        plt.xlabel('Class Interval')
        plt.ylabel('Frequency')
        plt.title('Histogram of Triangle Areas')
        plt.bar(x=x, height=height)
        plt.savefig(output_path)

    # Methods related with problem 2
    def unit_affine_transformation(self,
                                   affine: Dict = None,
                                   tri: List = None) -> List[List[int]]:
        transformed_tri = []
        matrix = affine['matrix']
        bias = affine['bias']
        for c in tri:
            transformed_vec = []
            for r in matrix:
                transformed_vec.append(r[0] * c[0] + r[1] * c[1])
            transformed_vec = [transformed_vec[0] + bias[0], transformed_vec[1] + bias[1]]
            transformed_tri.append(transformed_vec)
        return transformed_tri

    def dataset_affine_transformation(self,
                                      affine: Dict = None,
                                      dataset: Dict = None) -> Dict[int, List]:
        logger.info(f'Execute affine transformation')
        transformed_dataset = defaultdict(list)
        for n, t in dataset.items():
            transformed_dataset[n] = self.unit_affine_transformation(affine=affine,
                                                                     tri=t)
        return transformed_dataset

    def plot_triangle(self,
                      dataset: Dict = None) -> PatchCollection:
        patches = []
        for t in dataset.values():
            triangle = Polygon(np.array(t), True)
            patches.append(triangle)
        p = PatchCollection(patches,
                            cmap=matplotlib.cm.jet,
                            alpha=0.4)
        colors = 100 * np.random.rand(len(patches))
        p.set_array(np.array(colors))
        return p

    def plot_multi_dataset(self,
                           original: Dict = None,
                           transformed: Dict = None,
                           output_path: str = 'second.png') -> None:
        logger.info(f'Plot triangles before and after affine transformation to {output_path}')
        fig, ax = plt.subplots(2)

        op = self.plot_triangle(dataset=original)
        tp = self.plot_triangle(dataset=transformed)

        ax[0].set_title('Before affine transformation')
        ax[1].set_title('After affine transformation')

        omax = np.array(list(original.values())).max()
        tmax = np.array(list(transformed.values())).max()

        ax[0].set_xlim(0, omax)
        ax[0].set_ylim(0, omax)
        ax[1].set_xlim(0, tmax)
        ax[1].set_ylim(0, tmax)

        ax[0].add_collection(op)
        ax[1].add_collection(tp)
        plt.savefig(output_path)


def main():
    # Initiate instance
    operator = Operator2dTriangle(input_path=args.input_path,
                                  output_path=args.output_path,
                                  plot1_output_path=args.plot1_output_path,
                                  plot2_output_path=args.plot2_output_path)

    # Problem 1
    areas = operator.get_areas(dataset=operator.dataset[0])
    areas_mean = operator.get_areas_mean(dataset=areas)
    areas_std = operator.get_areas_std(dataset=areas,
                                       mean=areas_mean)
    histogram = operator.get_histogram(dataset=areas,
                                       n_bins=args.num_bins)
    operator.plot_histogram(histogram=histogram,
                            output_path=operator.plot1_output_path)

    # Problem 2
    transformed_dataset = operator.dataset_affine_transformation(affine=operator.dataset[1],
                                                                 dataset=operator.dataset[0])
    operator.plot_multi_dataset(original=operator.dataset[0],
                                transformed=transformed_dataset,
                                output_path=operator.plot2_output_path)



if __name__ == '__main__':
    main()
