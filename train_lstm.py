import torch
import argparse
import squad_data
import os

from torchmetrics import MetricCollection, Accuracy, F1, ConfusionMatrix

from torch import nn
from tqdm import tqdm

class ReasonLSTM(nn.Module):
    
    def __init__(self, data, emb_dim, hidden_dim, fc_dim, num_layers, dropout):
        super().__init__()
        self.device = data.device
        self.data = data
        self.embedding = nn.Embedding(len(data.vocab), 
                                      embedding_dim=emb_dim, 
                                      padding_idx=data.vocab.stoi['<pad>'])
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(in_features=hidden_dim*2, out_features=fc_dim)
        self.fc2 = nn.Linear(in_features=fc_dim, out_features=len(data.targets))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.to(self.device)
        print('\n\nInitialized LSTM\n\n')
        print(self)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(self.dropout(x))
        x = self.fc1(self.relu(x[:,-1,:]))
        x = self.fc2(self.dropout(x))
        return x
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path='./saves/model'):
        self.load_state_dict(torch.load(path))
    
    
class Training:
    
    def __init__(self, model, lr, epochs, sampler=False):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.softmax = nn.Softmax()
        if sampler:
            self.loss = nn.CrossEntropyLoss(weight=self.model.data.weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.metrics = MetricCollection({
            'accuracy' : Accuracy(),
            'f1' : F1(num_classes=len(model.data.targets), average='macro'),
            'cm' : ConfusionMatrix(num_classes=len(model.data.targets))
        })
    
    def train_model(self, save_path='./saves/model'):
        print(f'\n\nLSTM TRAINING USING : {self.model.device}\n\n\tlearning rate : {self.lr}\n\tepochs :\
            {self.epochs}\n\tloss : CrossEntropy\n\toptimizer : Adam\n\n')
        best_acc, best_ep = 0.0, 0
        self.model.train()
        for ep in range(self.epochs):
            ep = ep + 1
            print(f'\nTraining at epoch {ep}\n')
            self.model.train()
            for x, y in tqdm(self.model.data.train_iter):
                self.optimizer.zero_grad()
                predictions = self.model(x)
                loss_output = self.loss(predictions, y)
                loss_output.backward()
                self.optimizer.step()
            ep_accuracy = self.dev_metrics(ep)
            if ep_accuracy > best_acc:
                print(f'\n\n === new best accuracy {ep_accuracy} at epoch {ep} ===\n\n')
                best_acc = ep_accuracy
                best_ep = ep
                model.save_model(save_path)
        print(f'\n\nBEST ACCURACY : {best_acc} BEST EPOCH : {best_ep}\n\n')
        print(f'saved_model at: {save_path}')
        return best_acc
    
    def dev_accuracy(self, epoch):
        print(f'\n\nEvaluating at epoch : {epoch}\n')
        with torch.no_grad():
            self.model.eval()
            correct, all = 0, 0
            for x, y in tqdm(self.model.data.dev_iter):
                predictions = self.softmax(self.model(x))
                results = [torch.argmax(vec) for vec in predictions]
                for y_out, y_hat in zip(results, y):
                    if y_out == y_hat:
                        correct += 1
                    all += 1
        accuracy = correct / all
        print(f'\n\tAccuracy at epoch {epoch} : {accuracy}\n')
        return accuracy
    
    def dev_metrics(self, epoch):
        print(f'\n\nEvaluating at epoch : {epoch}\n')
        with torch.no_grad():
            self.model.eval()
            self.metrics.reset()
            for x, y in tqdm(self.model.data.dev_iter):
                predictions = self.model(x)
                self.metrics(predictions, y)
        report = self.metrics.compute()
        print(f"\n\tAccuracy at epoch {epoch} : {report['accuracy']}")
        print(f"\tF1 at epoch {epoch} : {report['f1']}\n")
        print(f"\tConfusion Matrix at epoch {epoch} : {report['cm']}")
        return report['accuracy']

class Test:
    
    def __init__(self, model):
        self.model = model
        self.metrics = MetricCollection({
            'accuracy' : Accuracy(),
            'f1' : F1(num_classes=len(model.data.targets), average='macro'),
            'cm' : ConfusionMatrix(num_classes=len(model.data.targets))
        })
    
    def test_model(self):
        print(f'\n\nTESTING MODEL\n\n')
        with torch.no_grad():
            self.model.eval()
            self.metrics.reset()
            for x, y in tqdm(self.model.data.test_iter):
                predictions = self.model(x)
                self.metrics(predictions, y)
        report = self.metrics.compute()
        print(f"\n\tAccuracy  : {report['accuracy']}")
        print(f"\tF1 : {report['f1']}\n")
        print(f"\tConfusion Matrix : {report['cm']}")
        return report['accuracy']
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='./data/shuffled_data/binary/train_debug.csv', help='path to training data')
    parser.add_argument('--test', type=str, default='./data/annotation_results/pos_tagged/arc_fc_th2_bin.csv', help='path to test data')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained')
    parser.add_argument('--max_vocab', type=int, default=20000, help='maximum vocabulary size')
    parser.add_argument('--min_freq', type=int, default=2, help='minimum occurrence frequency of features')
    parser.add_argument('--emb_dim', type=int, default=150, help='embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=250, help='hidden dimension')
    parser.add_argument('--fc_dim', type=int, default=150, help='linear dimension')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--gpu', type=lambda x: x in ['YES', 'yes', '1', 'True', 'true'], default=False, help='GPU available?')
    parser.add_argument('--split_sym', type=str, default=' ', help='how to tokenize the csv data')
    parser.add_argument('--sampler', type=lambda x: x in ['YES', 'yes', '1', 'True', 'true'] , default=False, help='weight samples for imbalanced dataset')
    args = parser.parse_args()
    
    data = squad_data.CSVProcessor(gpu=args.gpu,
                        train=args.train,
                        test=args.test,
                        max_size=args.max_vocab,
                        min_freq=args.min_freq,
                        batch_size=args.batch_size,
                        split_sym=args.split_sym,
                        sampler=args.sampler)
    
    model = ReasonLSTM(data=data,
                         emb_dim=args.emb_dim,
                         hidden_dim=args.hidden_dim,
                         fc_dim=args.fc_dim,
                         num_layers=args.num_layers,
                         dropout=args.dropout)
    
    if args.pretrained:
        model.load_model(args.pretrained)
        test = Test(model=model)
        test.test_model()
        
        
    else:
        trainer = Training(model=model, lr=args.lr, epochs=args.epochs, sampler=args.sampler)
        trainer.train_model()
        test = Test(model=model)
        test.test_model()
        
        