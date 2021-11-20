import configparser

config = configparser.ConfigParser()

with open('setup.cfg') as f:
    config.read_file(f)

metadata = config['metadata']

import argparse

parser = argparse.ArgumentParser(description='Return setup.cfg metadata value for given metadata key')
parser.add_argument('--key', metavar='key', nargs=1,
                    help='metadata key')
args = vars(parser.parse_args())

print(args)

if type(args['key']) == list:
    key = args['key'][0]

print(metadata[key])