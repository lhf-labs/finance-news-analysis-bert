# FinEAS: Financial Embedding Analysis of Sentiment 📈
(SentenceBERT for Financial News Sentiment Regression)

This repository contains the code for generating three models for Finance News Sentiment Analysis.

The models implemented are:
- A SentenceBERT with simple classifier.
- A BERT with simple classifier.
- A simple Bi-directional Long-Short Term Memory (LSTM) network.
- FinBERT from HuggingFace.
- SentenceBERT from HuggingFace

## Models 🤖
- FinEAS: https://huggingface.co/LHF/FinEAS *(not available)*
- FinBERT: https://huggingface.co/LHF/finbert-regressor *(not available)*

## Results ✅
We used three partitions of the datasets from the February 11th, 2021. 6 months previous to that date,
1 year previous to that date and 2 years previous to the date mentioned.

We also evaluated the models 2 weeks later that date; that is to say, we evaluated from February 12th, 2021
to February 26th, 2021.

The table below shows the results:

|                          | FinEAS | BERT   | BiLSTM |
|--------------------------|--------------|--------|--------|
| 6 months                 |   **0.0556** | 0.2124 | 0.2108 |
| 6 months<br>Next 2 weeks |   **0.1061** | 0.2190 | 0.2194 |
| 1 year                   |   **0.0654** | 0.2137 | 0.2140 |
| 1 year<br>Next 2 weeks   |   **0.1058** | 0.2191 | 0.2194 |
| 2 years                  |   **0.0671** | 0.2087 | 0.2086 |
| 2 years<br>Next 2 weeks  |   **0.1065** | 0.2188 | 0.2185 |


The table below shows the results for the HuggingFace models
| Dates     | FinEAS | FinBERT |
|-----------|--------|---------|
| 6 months  | 0.0044 | 0.0050  |
| 12 months | 0.0036 | 0.0034  |
| 24 months | 0.0033 | 0.0040  |


## Citing 📣
```
@misc{gutierrezfandino2021fineas,
      title={FinEAS: Financial Embedding Analysis of Sentiment}, 
      author={Asier Gutiérrez-Fandiño and Miquel Noguer i Alonso and Petter Kolm and Jordi Armengol-Estapé},
      year={2021},
      eprint={2111.00526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License 🤝
MIT License.

Copyright 2021 Asier Gutiérrez-Fandiño & Jordi Armengol-Estapé.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
