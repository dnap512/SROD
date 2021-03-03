import torch

import utility
import data
import model
import loss
import numpy as np
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
           
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            model_parameters = filter(lambda p: p.requires_grad, _model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("the number of model's parameters :",params)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
