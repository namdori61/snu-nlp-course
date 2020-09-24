from glob import glob
import json
from tqdm import tqdm
import csv
import logging
import argparse

parser = argparse.ArgumentParser(description='Generate tsv format dataset.')
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
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerow(['filename', 'date', 'NewsPaper', 'Topic', 'Original_Topic', 'News'])
        for f in tqdm(file_list, desc='file processing'):
            filename = f.split('/')[-1][:-5]
            with open(f, 'r') as input_file:
                data = json.load(input_file)
                for d in data['document']:
                    date = d['metadata']['date']
                    publisher = d['metadata']['publisher']
                    topic = d['metadata']['topic']
                    original_topic = d['metadata']['original_topic']
                    for p in d['paragraph']:
                        text = p['form'].strip('"')
                        writer.writerow([filename, date, publisher, topic, original_topic, '<p>' + text + '</p>'])
                        sent_cnt += 1
    logger.info(f'Total {sent_cnt} text processed.')


if __name__ == '__main__':
    main()
