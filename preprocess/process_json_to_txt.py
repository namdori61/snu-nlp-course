from glob import glob
import json
from tqdm import tqdm
import logging
import argparse

parser = argparse.ArgumentParser(description='Generate input dataset.')
parser.add_argument('-i', '--input_dir', type=str, default=None,
                    help='Directory of files to process.')
parser.add_argument('-o', '--output_path', type=str, default=None,
                    help='Output file path to save processed dataset.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = parser.parse_args()


def main():
    logger.info(f'Read files from {args.input_dir}')
    file_list = glob(args.input_dir + '/*.json')

    logger.info('Start processing!')
    sent_cnt = 0
    with open(args.output_path, 'w') as output_file:
        for f in tqdm(file_list, desc='file processing'):
            with open(f, 'r') as input_file:
                data = json.load(input_file)
                for d in data['document']:
                    for p in d['paragraph']:
                        text = p['form'].strip('"')
                        output_file.write('<p> ' + text + ' </p> \n')
                        sent_cnt += 1
    logger.info(f'Total {sent_cnt} text processed.')


if __name__ == '__main__':
    main()
