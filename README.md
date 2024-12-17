# Large Language Models for Medical Forecasting - Foresight 2

**Paper: Under review**

Foresight 2 (FS2) is a large language model fine-tuned on hospital data for modelling patient timelines. 
It can understand patients’ clinical notes and predict SNOMED codes for a wide range of biomedical use cases,
including diagnosis suggestions, risk forecasting, and procedure and medication
recommendations. FS2 is trained on the free text portion of the MIMIC-III dataset,
firstly through extracting biomedical concepts and then creating contextualised
patient timelines, upon which the model is then fine-tuned. The results show
significant improvement over the previous state-of-the-art for the next new biomedical 
concept prediction (P/R - 0.73/0.66 vs 0.52/0.32) and a similar improvement
specifically for the next new disorder prediction (P/R - 0.69/0.62 vs 0.46/0.25).
Finally, on the task of risk forecast, we compare our model to GPT-4-turbo (and a
range of open-source biomedical LLMs) and show that FS2 performs significantly
better on such tasks (P@5 - 0.90 vs 0.65). This highlights the need to incorporate
hospital data into LLMs and shows that small models outperform much larger ones
when fine-tuned on high-quality, specialised data.

## Data Preparation
<img src="https://github.com/w-is-h/fs2/blob/main/media/architecture.png" width=400px/>

