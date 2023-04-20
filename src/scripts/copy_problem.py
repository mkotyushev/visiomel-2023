import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='Path to data folder')
parser.add_argument('--output-dir', type=Path, help='Path to output folder')
args = parser.parse_args()

problem_filenames = ['1vig9enh.png', '24xxi0a4.png', '3f4c0t1d.png', '61wseljn.png', '6235nfns.png', '6gt1vaty.png', '7nyiq3xw.png', '8f7dpkpv.png', '8j2n6c96.png', '8pkdid6b.png', '9d9fv6qj.png', '9dp3he0s.png', '9r67fy7w.png', 'aw75mfil.png', 'ba9kpgk3.png', 'dzuwx6cz.png', 'evlwfw6j.png', 'f54rwukl.png', 'g64ajigb.png', 'hi63pfrb.png', 'hxiubwtq.png', 'i5h28l20.png', 'l6s043vm.png', 'ltwbgd9g.png', 'mut49okk.png', 'mwr9rmlg.png', 'onxdf2fq.png', 'oxk5y262.png', 'peesbo3q.png', 'pkfqj0ga.png', 'qrimite8.png', 'rdslu5i5.png', 'rr4gq7zb.png', 's4cm9bva.png', 'uuggf1s7.png', 'vif6zvk6.png', 'z9o9gqia.png']

for filepath in Path(args.input_dir).glob('**/*.png'):
    if filepath.name not in problem_filenames:
        continue

    output_filepath = args.output_dir / filepath.relative_to(args.input_dir)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(filepath, output_filepath)
