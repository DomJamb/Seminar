import argparse
from pathlib import Path
from data.cityscapes import IBAPoisonCityscapes


def main(args):
    for split in args.splits:
        dataset = IBAPoisonCityscapes(
            Path(args.dataset_root),
            subset=split,
            transforms=lambda x: x,
            poison_type=args.poison_type,
            poisoning_rate=args.poisoning_rate,
            cached_root=Path(args.cached_root),
            resize_size=args.resize_size,
            trigger_size=args.trigger_size,
            victim_class=args.victim_class,
            target_class=args.target_class,
        )
        print(len(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='./datasets/cityscapes')
    parser.add_argument('--cached_root', type=str, default='./cached_data')
    parser.add_argument('--resize_size', type=tuple, default=(1024, 512))
    parser.add_argument('--trigger_size', type=int, default=55)
    parser.add_argument('--victim_class', type=str, default='car')
    parser.add_argument('--target_class', type=str, default='road')
    parser.add_argument("--poisoning_rate", type=float, default=0.2)
    parser.add_argument('--poison_type', type=str, default='IBA')
    parser.add_argument('--splits', nargs='+', default=['train', 'val_poisoned', 'val'])
    main(parser.parse_args())