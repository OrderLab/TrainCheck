import argparse

from traincheck.invariant import read_inv_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="get the number of invariants for each relation"
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="+",
        required=True,
        help="Invariants files to know the number of invariants for each relation",
    )

    args = parser.parse_args()
    for inv_file in args.invariants:
        invs = read_inv_file(inv_file)
        print(f"Number of invariants for {inv_file}: {len(invs)}")
        print("Number of invariants for each relation:")
        relation_count = {}
        for inv in invs:
            relation = inv.relation
            if relation not in relation_count:
                relation_count[relation] = 0
            relation_count[relation] += 1
        for relation, count in relation_count.items():
            print(f"  {relation}: {count}")
        print()
