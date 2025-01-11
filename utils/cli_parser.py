import argparse

def parse_args(known_args:dict):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model', type=str, choices=['u-net', 'u-resnet34', 'u-net++', 'deeplabv3+', 'sam', 'sam_lora'])
    parser.add_argument('--sam_pretrain_weights', type=str)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dataset', type=str, choices=['PA', 'PA284', 'traffic'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_workers', type=int)

    cli_args = parser.parse_args()
    cli_args = vars(cli_args)
    cli_args = {k: v for k, v in cli_args.items() if v is not None}
    known_args.update(cli_args)

