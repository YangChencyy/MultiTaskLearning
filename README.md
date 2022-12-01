# MultiTaskLearning
Team Member: Yang Chem, Run Peng, Yang Fei
<br> 
Final Project for EECS 498-10, instructor: Prof. Emily Provost

## Project Description
Depression is a common illness worldwide, with an estimated 3.8% of the population affected according to WHO. 
Depressed people usually find it hard to communicate with people, concentrate on work and may even have the desire to suicide. 
Hence the detection and treatment of depression becomes a major issue nowadays. 
However, current techniques of depression detection are mainly based on clinicians’ review, or patients’ self-reports, which risks a range of subjective biases and over-consumption of time.
Therefore, we design a multi-modal depression detection model via deep learning that can smartly detect the level of depression for patients. 
This model is aimed to provide an objective, professional, fast and accurate reference for doctors’ diagnosis. 
<br> 
<br> 
The multi-modal model can take in both audio recordings and text-based transcripts of dialogues between patients and therapists as inputs, providing more information for the decision making. 
This is a novelty compared with the traditional model which considers only one source of input. 
Also, it will perform two tasks parallel within the model to provide two predictions, a binary prediction (positive or negative) for whether the patient has depression, and a quantified PHQ-8 score, which reflects the severity of a patient’s level of depression. 
The application of multi-class learning will improve the model’s performance for predictions. 
Moreover, with the predicted model, prediction for one sample of input can be generated in a relatively little time. 
From these, we believe our product will perform well on a technical level and be accepted by healthcare professionals in depression learning areas.

## Some Notes
Check our final reports in report -> Multi-Modal Depression Detection via Deep Learning.pdf.

If you want to run the code or interested in reproduce some results, you may need to contact boberg@ict.usc.edu from University of Southern California for access of the dataset.
