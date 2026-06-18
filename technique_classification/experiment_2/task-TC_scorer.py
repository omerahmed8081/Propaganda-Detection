import sys
import argparse
import logging.handlers
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import src.annotation as an
import src.annotations as ans
import src.propaganda_techniques as pt

logger = logging.getLogger("propaganda_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)


def main(args):

    user_submission_file = args.submission
    gold_file = args.gold

    # --------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------
    def load_file(path):
        data = {}
        all_labels = set()

        with open(path, "r") as f:
            for line in f:
                article, label, start, end = line.strip().split("\t")
                key = (article, int(start), int(end))

                if key not in data:
                    data[key] = set()

                data[key].add(label)
                all_labels.add(label)

        return data, all_labels

    pred, pred_labels = load_file(user_submission_file)
    gold, gold_labels = load_file(gold_file)

    all_keys = set(gold.keys()) | set(pred.keys())
    all_labels = sorted(list(pred_labels | gold_labels))

    # --------------------------------------------------
    # GLOBAL METRICS
    # --------------------------------------------------
    tp = 0
    fp = 0
    fn = 0

    for key in all_keys:
        gold_set = gold.get(key, set())
        pred_set = pred.get(key, set())

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n🚀 RELAXED SCORING RESULTS")
    print(f"Precision={precision:.6f}")
    print(f"Recall={recall:.6f}")
    print(f"F1={f1:.6f}")

    # --------------------------------------------------
    # PER-CLASS METRICS
    # --------------------------------------------------
    print("\n📊 Per-class F1:")

    for label in all_labels:
        tp = fp = fn = 0

        for key in all_keys:
            gold_set = gold.get(key, set())
            pred_set = pred.get(key, set())

            if label in gold_set and label in pred_set:
                tp += 1
            elif label not in gold_set and label in pred_set:
                fp += 1
            elif label in gold_set and label not in pred_set:
                fn += 1

        precision_cls = tp / (tp + fp + 1e-6)
        recall_cls = tp / (tp + fn + 1e-6)
        f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-6)

        print(f"F1_{label}={f1_cls:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Scorer for SemEval 2020 Task 11 subtask TC.\n" +
    "Example: python3 task-TC_scorer.py -s data/submission-task-TC.tsv -r data/article736757214.task-FLC.labels -p data/propaganda-techniques-names-semeval2020task11.txt")

    parser.add_argument('-s', '--submission-file', dest='submission', required=True, help="file with the submission of the team")
    parser.add_argument('-r', '--reference-file', dest='gold', required=True, help="file with the gold labels.")
    parser.add_argument('-d', '--enable-debug-on-standard-output', dest='debug_on_std', required=False,
                        action='store_true', help="Print debug info also on standard output.")
    parser.add_argument('-l', '--log-file', dest='log_file', required=False, help="Output logger file.")
    parser.add_argument('-p', '--propaganda-techniques-list-file', dest='propaganda_techniques_list_file', required=True, 
                        help="file with list of propaganda techniques (one per line).")
    parser.add_argument('-o', '--output-for-script', dest='output_for_script', required=False, action='store_true',
                        default=False, help="Prints the output in a format easy to parse for a script")
    main(parser.parse_args())
