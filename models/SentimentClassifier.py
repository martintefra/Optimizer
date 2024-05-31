import torch.nn as nn
from transformers import BertModel



class SentimentClassifier(nn.Module):
    """
    A sentiment classifier that uses a pre-trained BERT model.

    Args:
        num_classes (int): The number of sentiment classes.

    Attributes:
        bert (BertModel): The pre-trained BERT model.
        dropout (nn.Dropout): A dropout layer.
        linear (nn.Linear): A linear layer that maps the BERT output to the sentiment classes.

    Methods:
        forward(input_ids, attention_mask, token_type_ids): Forward pass of the sentiment classifier.
    """
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Pass the inputs to the model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Extract the last hidden state of the first token of the sequence
        pooled_output = outputs[1]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Pass the output through a linear layer to get the logits
        logits = self.linear(pooled_output)

        return logits