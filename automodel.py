import torch
import torch.nn as nn
import torch.optim as optim



import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle

########################################################################################################################

model_name="google/muril-base-cased"
lr=2e-5
n_epochs=5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)


########################################################################################################################


def clean_tweet(tweet):
    #tweet = re.sub(r"@[A-Za-z0-9]+",' ', tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]+",' <url> ', tweet)
    tweet = re.sub(r" +", ' ', tweet)
    return tweet
def get_data_val(X, y, device, batch_size=512):
    assert X.shape[0] == y.shape[0]
    data_length = X.shape[0]
    no_of_batches = (data_length // batch_size)+(data_length % batch_size!=0)
    shuffled_idxs = np.random.permutation(list(range(data_length)))

    for i in range(no_of_batches):
        idxs = shuffled_idxs[i*batch_size:i*batch_size+batch_size]
        yield(X[idxs],y[idxs])
        #yield torch.IntTensor(X[idxs]).to(device),\
                #torch.LongTensor(y[idxs]).to(device), batch_size
def get_data_train(X, y, device, batch_size=512):
    assert X.shape[0] == y.shape[0]
    batch_size_=batch_size//2
    X_1=X[y==0]
    X_2=X[y==1]


    data_length = min(X_1.shape[0],X_2.shape[0])
    no_of_batches = (data_length // batch_size)+(data_length % batch_size!=0)
    shuffled_idxs = np.random.permutation(list(range(data_length)))

    for i in range(no_of_batches):
        idxs = shuffled_idxs[i*batch_size:i*batch_size+batch_size]
        yield(X_1[idxs].tolist()+X_2[idxs].tolist(),np.array([0]*batch_size+[1]*batch_size))
        #yield torch.IntTensor(X[idxs]).to(device),\
                #torch.LongTensor(y[idxs]).to(device), batch_size

class BertBinaryClassifier(nn.Module):
    def __init__(self):
        super(BertBinaryClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 2)

        # self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        #self.tanh
        
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.bert(x, x!=0).pooler_output
        x = self.linear(x)
        #x = self.softmax(x)
        return x

########################################################################################################################
with open("./data/train.p","rb") as f:
    data=pickle.load(f)
texts=[i[1] for i in data]
for i in data:
    try:
        int(i[-1])
    except:
        print(i)
y=[int(i[-1]) for i in data]

'''
tokens = tokenizer(texts)["input_ids"]
length = 500
h = [0]*((length//10)+1)
for x in tokens:
    h[len(x)//10 if len(x) < length else length//10] += 1
plt.bar(list(range((length//10)+1)), h)
plt.show()
max_tokens_len=max([len(i) for i in tokens])
print("Max_length: ",max_tokens_len)
max_tokens_len=int(np.percentile([len(i) for i in tokens],99))
'''
max_tokens_len=64

#X = tokenizer(texts,max_length=max_tokens_len,padding=True)["input_ids"]
#X.shape
p=np.mean(y)
print("% of 1:",p)
x_train, x_val, y_train, y_val = train_test_split(np.arange(len(texts)), np.array(y), test_size=0.2, random_state=42)
x_train.shape, x_val.shape, y_train.shape, y_val.shape

from transformers import get_linear_schedule_with_warmup
model = BertBinaryClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss(torch.Tensor([p,1-p]).to(device))
batch_size=16
grad_acc_num=2
num_training_steps=n_epochs*len(x_train)//(batch_size*grad_acc_num)
num_warmup_steps=int(0.1*num_training_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_training_steps
)



for epoch_num in range(n_epochs):
    print("--------- EPOCH ", epoch_num + 1, "---------")
    print("Training:")
    model.train()
    train_y, train_y_hats = [], []
    for step_num, batch_data in tqdm(enumerate(get_data_val(x_train, y_train,
                                                        device, batch_size)),
                                     total=(len(x_train)//batch_size)+1):
        X, y = [texts[ijj] for ijj in batch_data[0]],  torch.LongTensor(batch_data[1]).to(device)
        X = tokenizer(X,max_length=max_tokens_len,padding=True,truncation=True,return_tensors='pt')["input_ids"].to(device)
        #print(X.shape)
        y_hat = model(X)
        
        
        batch_loss = loss_func(y_hat, y)
        batch_loss.backward()
        if(step_num%grad_acc_num==0):
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        train_y.extend(batch_data[1])
        train_y_hats.extend([y2 for y2 in y_hat.argmax(1).cpu().numpy()])
        if(step_num%1000==0):
            
            print()
            print(classification_report(train_y, train_y_hats))
            train_y, train_y_hats = [], []

    print("Validating:")
    model.eval()
    val_y, val_y_hats = [], []
    torch.save(model.to("cpu"), f'./saved_models/model_{epoch_num}_{step_num}.pth')
    for step_num, batch_data in tqdm(enumerate(get_data_val(x_val, y_val,
                                                        device, batch_size)),
                                     total=(len(x_val)//batch_size)+1):
        with torch.no_grad():
            X, y = [texts[ijj] for ijj in batch_data[0]],  torch.LongTensor(batch_data[1]).to(device)
            X = tokenizer(X,max_length=max_tokens_len,padding=True,truncation=True, return_tensors='pt')["input_ids"].to(device)
            y_hat = model(X)
            val_y.extend(batch_data[1])
            val_y_hats.extend([y2 for y2 in y_hat.argmax(1).cpu().numpy()])
    print()
    print(classification_report(val_y, val_y_hats))

