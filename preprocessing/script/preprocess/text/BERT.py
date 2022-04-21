import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dir_path = "Transcript/"
paths = os.listdir(dir_path)
sentence_embedding_path = "BERT/sentence_embedding/"
word_embedding_path = "BERT/word_embedding/"
for path in paths:
    print(path)
    df = pd.read_csv(dir_path + path)
    text = ' '.join(df["Text"].tolist())
    marked_text =  "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)[:512]
    if len(tokenized_text) == 512:
        tokenized_text[511] = "[SEP]"


    # print(len(tokenized_text))
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


    # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        
        hidden_states = outputs[2]
        pooling_outputs = outputs[1]

    # token_vecs = hidden_states[-2][0]
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)

    # # Calculate the average of all 22 token vectors.
    # sentence_embedding = torch.mean(token_vecs, dim=0)
    # print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
    torch.save(outputs[1], sentence_embedding_path + '{}_sentence_embedding.pt'.format(path[:3]))
    torch.save(outputs[0], word_embedding_path + '{}_word_embedding.pt'.format(path[:3]))