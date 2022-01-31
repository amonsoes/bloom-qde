# Bloom QDE

This repository stores the code, data and experiments for Bloom QDE. This is the accompanying repository to the paper *put paper here*

```bash
pip install -r requirements.txt
```

## Reproduce Experiment Results

### PoS-Tagging

- Takes an input CSV in the format question,class (EASY,DIFFICULT) and outputs a csv where all words are PoS tagged

- Note: Question words are left as they are since the contribute to the question difficulty

```preprocessing.py --infile='new_data_annotation_results_second_choice_th2_binary.csv' --outfile='arc_masked_annotation_results_second_choice_th2_binary.csv'
```

## Build Dataset from Custom Data

## Train Customized Model for QDE

