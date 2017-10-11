import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from lib.utils import *


def predict(file_paths, opened_files=[], coefficients=None):
    """
    Creates new submission by merging list of existing ones with given weights.

    :param file_paths: paths to submissions which will be merged
    :param coefficients: weights for merging submissions
    """

    if len(opened_files) < len(file_paths):
        print("Open %s" % file_paths[len(opened_files)])
        with open(file_paths[len(opened_files)], 'r') as f:
            opened_files.append(f)
            yield from predict(file_paths, opened_files, coefficients)
    else:
        n = 0
        if coefficients is None:
            coefficients = [1.0] * len(opened_files)
        for lines in zip(*opened_files):
            if n % 100 == 0:
                print(n)
            if n == 0:
                n += 1
                continue
            else:
                n += 1
            names_with_rles = [tuple(x.strip().split(',')) for x in lines]
            for i in range(1, len(names_with_rles)):
                assert names_with_rles[0][0] == names_with_rles[i][0]
            masks = [rle_decode(x[1], INIT_HEIGHT, INIT_WIDTH) for x in names_with_rles]
            masks = [x[0] * x[1] for x in zip(masks, coefficients)]
            res = np.sum(masks, 0) / np.sum(coefficients) > 0.5
            yield (names_with_rles[0][0], res)


def run_predict():
    submissions = [
        os.path.join("../submissions", submission)
        for submission in [
            "sample_submission.csv"
        ]
    ]
    weights = [1.0]
    create_submission(predict(submissions, coefficients=weights))


if __name__ == '__main__':
    run_predict()
    print('Success!')
