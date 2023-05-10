# Extract 'state_dict' from all "**/epoch=*.ckpt" files, save it with torch.save() 
# to a file with the same name as the "**/epoch=*.ckpt" file but 
# with the extension changed to ".pth". Keep the same directory structure.
#
# Use pathlib.Path.glob() to find all "**/epoch=*.ckpt" files.
#
# args: input-dir, output-dir
# input-dir: directory containing checkpoints
# output-dir: directory to save weights
# example: python src/scripts/convert_checkpoints_to_weights.py ./visiomel ./weights/visiomel

import argparse
import pathlib
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing checkpoints")
    parser.add_argument("output_dir", type=str, help="directory to save weights")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    for checkpoint_path in input_dir.glob("**/epoch=*.ckpt"):
        output_path = output_dir / checkpoint_path.relative_to(input_dir).with_suffix(".pth")
        print(f'Converting {checkpoint_path} to {output_path}')
        state_dict = torch.load(checkpoint_path)["state_dict"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, output_path)


if __name__ == "__main__":
    main()
