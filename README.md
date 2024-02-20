# RSA-GUI

## Resources
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2605405/

## Description
RSA GUI evaluates two different systems that are representing information, i.e. stimuli.
Input: Two RDMs
Output: Spearman Correlation Coefficient (for now)

## How to run
If it is the first time running
```
pip3 install -r requirements.txt
conda create -n thingsvision python=3.9
conda activate thingsvision
pip3 install --upgrade thingsvision
pip3 install git+https://github.com/openai/CLIP.git
Python3 combined_gui.py
```

If it is not the first time running
```
conda activate thingsvision
Python3 combined_gui.py
```