import argparse
from collections import defaultdict
from itertools import combinations
from typing import *

ignore_set = ['FEMALE', 'MALE', '<18', '18-24', '25-44', '45-64', '65+']

def get_arguments():
    # Used to limit the choices for the precision entered by the user
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, x):
            return self.start <= x <= self.end

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Add required positional arguments
    parser.add_argument(
        'f',
        help='Path to the csv data file',
        metavar='<f>',
        type=str,
    )
    parser.add_argument(
        'min_sup',
        type=float,
        choices=[Range(0.0, 1.0)],
        help='The minimum support',
        metavar='min sup'
    )
    parser.add_argument(
        'min_conf',
        type=float,
        choices=[Range(0.0, 1.0)],
        help='The minimum confidence',
        metavar='min conf'
    )

    # Parse the arguments
    return parser.parse_args()


# Load transactions from the integrated dataset and find the frequencies of the initial item sets (all with size 1)
def read_csv(f: str) -> Tuple[List[Set[str]], Dict[FrozenSet[str], int]]:
    transactions = []
    item_counts = defaultdict(int)
    with open(f, 'r') as f:
        for line in f:
            transaction = set(line.strip().split(','))
            for item in transaction:
                item_counts[frozenset({item})] += 1
            transactions.append(transaction)
    return transactions, item_counts


# Algorithm in Sect. 2.1.1 of the paper
def generate_candidates(l_set: List[Tuple[FrozenSet[str], float]], k: int) -> Set[FrozenSet[str]]:
    # Join step
    raw_c_set = set()
    for i in range(len(l_set)):
        p = sorted(l_set[i][0])
        for j in range(len(l_set)):
            q = sorted(l_set[j][0])
            if p[:k - 2] == q[:k - 2] and p[k - 2] < q[k - 2]:
                # Take the union of p and the last element of q (sorted)
                raw_c_set.add(frozenset(l_set[i][0] | {q[k - 2]}))

    # Prune step
    l_set_s = set(tup[0] for tup in l_set)  # convert list of (set, support) to set of set
    c_set = set()
    for c in raw_c_set:
        # Add to the set iff all the subsets with size k - 1 are in the frequent set
        if all(frozenset(s) in l_set_s for s in combinations(c, k - 1)):
            c_set.add(c)
    return c_set


# Algorithm in Sect. 2.1 of the paper
def a_priori(
    transactions: List[Set[str]],
    item_counts: Dict[FrozenSet[str], int],
    min_sup: float
) -> List[Tuple[FrozenSet[str], float]]:
    n = float(len(transactions))

    # Process the initial item counts (item set size 1)
    l_set = []
    for item, cnt in item_counts.items():
        sup = cnt / n
        if sup >= min_sup:
            l_set.append((item, sup))

    # Iterate and find frequent item sets with greater sizes
    k = 2
    res = []
    while len(l_set):
        c_set = generate_candidates(l_set, k)

        tmp_count = defaultdict(int)
        for c in c_set:
            for t in transactions:
                if t.issuperset(c):
                    tmp_count[c] += 1
                    item_counts[c] += 1
        l_set = []
        for c, cnt in tmp_count.items():
            sup = cnt / n
            if sup >= min_sup:
                l_set.append((c, sup))

        res.extend(l_set)
        k += 1

    return res


# Build association rules from frequent item sets
def build_association_rules(
    item_sets: List[Tuple[FrozenSet[str], float]],
    item_counts: Dict[FrozenSet[str], int],
    min_conf: float
) -> List[Tuple[str, float, float]]:
    res = []
    for item_set, sup in item_sets:
        for s in combinations(item_set, len(item_set) - 1):  # only 1 item on RHS
            ss = frozenset(s)
            conf = item_counts[item_set] / item_counts[ss]
            if (conf >= min_conf) and (list(item_set - ss)[0] not in ignore_set):
                res.append((f"{sorted(s)} => {list(item_set - ss)}", conf, sup))
    return res


# Write item sets and rules to file
def write_output(
    args,
    item_sets: List[Tuple[FrozenSet[str], float]],
    rules: List[Tuple[str, float, float]],
    output_path: str = "output.txt"
) -> None:
    with open(output_path, 'w+') as f:
        f.write(f"==Frequent itemsets (min_sup={round(args.min_sup * 100, 4)}%)\n")
        for item_set, sup in item_sets:
            f.write(f"{list(item_set)}, {round(sup * 100, 4)}%\n")
        f.write(f"==High-confidence association rules (min_conf={round(args.min_conf * 100, 4)}%)\n")
        for rule, conf, sup in rules:
            f.write(f"{rule} (Conf: {round(conf * 100, 4)}%, Supp: {round(sup * 100, 4)}%)\n")


def main():
    args = get_arguments()
    transactions, item_counts = read_csv(args.f)
    frequent_item_sets = a_priori(transactions, item_counts, args.min_sup)
    frequent_item_sets.sort(key=lambda x: -x[1])  # sort by support DESC
    rules = build_association_rules(frequent_item_sets, item_counts, args.min_conf)
    rules.sort(key=lambda x: -x[1])  # sort by confidence DESC
    write_output(args, frequent_item_sets, rules)


if __name__ == '__main__':
    main()
