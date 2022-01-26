import torchtext
import torch

# antwortumfang und komplexität - mehrsatzantworten und herleitungen

class BatchWrapper:
    
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

      def __iter__(self):
            for batch in self.dl:
                x = getattr(batch, self.x_var)
                y = batch.label
                yield (x, y)

      def __len__(self):
            return len(self.dl)

class CSVProcessor:
    
    def __init__(self, gpu, train, dev, test, max_size, min_freq, batch_size, split_sym, sampler=False, train='train.csv', dev='dev.csv'):
      print('\n\nProcessing data...\n')
      self.device = 'cuda:0' if gpu == True else 'cpu'
      self.TEXT = torchtext.data.Field(lower=True, tokenize=lambda x: x.split(split_sym), use_vocab=True, batch_first=True)
      self.LABEL = torchtext.data.Field(sequential=False, is_target=True, batch_first=True, unk_token=None)
      fields = [('text', self.TEXT), ('label', self.LABEL)]
      """
      if test:
            dataset = torchtext.data.TabularDataset(path=datapath,
                                    fields=fields,
                                    format='csv',
                                    skip_header=True)
            self.train, self.test, self.dev = dataset.split(split_ratio=[0.8, 0.05, 0.15])
            self.LABEL.build_vocab(self.train, self.dev, self.test)
      """
      self.train, self.dev = torchtext.data.TabularDataset.splits(path=datapath,
                                                                        train=train,
                                                                        validation=dev,
                                                                        skip_header=True,
                                                                        format='csv',
                                                                        fields=fields)
      self.test = dataset = torchtext.data.TabularDataset(path=test,
                                    fields=fields,
                                    format='csv',
                                    skip_header=True)
      self.LABEL.build_vocab(self.train, self.dev, self.test)
      self.TEXT.build_vocab(self.train, max_size=max_size, min_freq=min_freq)
      self.targets = self.LABEL.vocab
      self.vocab = self.TEXT.vocab
      if sampler:
            weights = {cls:0 for cls in self.targets.stoi}
            for example in self.train:
                  weights[example.label] += 1
            lowest_val = weights[min(weights, key=weights.get)]
            weights = {k:(lowest_val/v) for k,v in weights.items()}
            self.weights = torch.tensor([weights[self.targets.itos[i]] for i in range(len(self.targets.stoi))])
      train_iter = torchtext.data.BucketIterator(self.train,
                                                batch_size,
                                                train=True,
                                                shuffle=True,
                                                sort_key=lambda x: len(x.text),
                                                sort_within_batch=True,
                                                device=self.device)
      dev_iter = torchtext.data.BucketIterator(self.dev, batch_size, sort_key=lambda x: len(x.text), device=self.device)
      test_iter = torchtext.data.BucketIterator(self.test, batch_size, sort_key=lambda x: len(x.text), device=self.device)
      self.train_iter = BatchWrapper(train_iter, "text", "label")
      self.dev_iter = BatchWrapper(dev_iter, "text", "label")
      self.test_iter = self.dev_iter = BatchWrapper(test_iter, "text", "label")
      print('\ndone\n\n')

if __name__ == '__main__':
      
      difficult = 0
      easy = 0

      gpu=False
      datapath = '/Users/amonsoares/Desktop/Question Complexity/data/data_binary.csv'
      data = CSVProcessor(gpu=gpu,
                        datapath=datapath,
                        max_size=10000,
                        min_freq=2,
                        batch_size=1,
                        split_sym=' ',
                        sampler=True)
    
        