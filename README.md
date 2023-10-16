### Credit to Original Repositories
- Original repo: https://github.com/Audio-AGI/AudioSep
- camenduru's colab with the gradio UI: https://github.com/camenduru/AudioSep-colab

This repo is forked from camenduru's colab code to install locally.

# Installation
1. Create new conda environment
```
conda create -n audiosep python=3.10
conda activate audiosep
```

2. Clone the repository
```
git clone https://github.com/natlamir/AudioSep.git
cd AudioSep
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Download required files
- [model weights](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint): place in the ```checkpoint``` folder
- [audioset_textmap.npy](https://github.com/natlamir/ProjectFiles/blob/main/AudioSep/audioset_textmap.npy): place in the ```models/CLAP/training``` folder
- [bpe_simple_vocab_16e6.txt.gz](https://github.com/natlamir/ProjectFiles/blob/main/AudioSep/bpe_simple_vocab_16e6.txt.gz): place in the ```models/CLAP/open_clip``` folder

# Usage
Run the gradio app with:
```
python app.py
```
