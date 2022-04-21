# EECS498-010 Project

# Data Preprocessing

For audio, we tried to use wav2vec model to generate rich embeddings for each audio file, but failed because of limited computational resources. Therefore, we choose to take the pre-generated mfcc embeddings, provided in the dataset by using OpenSmile 2.3.0, as features of audios.

However, the mfcc embeddings for the whole conversations are still too heavy for further computation. Therefore, we assume that the information of whole conversations is not necessary, and the partial data also contain meaningful information. Thus, we further filtered the datset with fixed size of output, where we use normal distribution as filtering base. Then we pass the filtered output to our prediction model, which largely boosts the efficiency, and make it avaiable to run on our machine.

The reason for why we use normal distribution, instead of uniform distribution is because: according to [1], useful information has higher probabilities to accumulate around the middle part of the whole conversations / texts. Therefore, normal distribution has its superiority over uniform distribution. To keep temporal information, we crop the audio embeddings with fixed steps, and then do the sampling.






[1] C. Sun, X. Qiu, Y. Xu, and X. Huang, “How to fine-tune bert for text classification?” Chinese Comput. Linguistics, vol. 11856, pp. 194–206, 2019