import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

from nets import GCN
from graphs import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'pubmed', 'citeseer'])
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=16,
                        help='Number of units')
    parser.add_argument('--dropout', '-d', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--validation-interval', type=int, default=1,
                        help='Number of updates before running validation')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping.')
    parser.add_argument('--normalization', default='gcn',
                        choices=['pygcn', 'gcn'],
                        help='Variant of adjacency matrix normalization method to use')
    args = parser.parse_args()

    print("Loading data")
    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset, normalization=args.normalization)

    train_iter = chainer.iterators.SerialIterator(
        idx_train, batch_size=len(idx_train), shuffle=False)
    dev_iter = chainer.iterators.SerialIterator(
        idx_val, batch_size=len(idx_val), repeat=False, shuffle=False)

    # Set up a neural network to train.
    print("Building model")
    model = GCN(adj, features, labels, args.unit, dropout=args.dropout)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    if args.weight_decay > 0.:
        optimizer.add_hook(
            chainer.optimizer_hooks.WeightDecay(args.weight_decay))

    if args.model != None:
        print("Loading model from " + args.model)
        chainer.serializers.load_npz(args.model, model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    if args.early_stopping:
        trigger = training.triggers.EarlyStoppingTrigger(
            monitor='validation/main/loss', patients=10,
            check_trigger=(args.validation_interval, 'epoch'),
            max_trigger=(args.epoch, 'epoch'))
    else:
        trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, trigger, out=args.out)

    trainer.extend(extensions.Evaluator(dev_iter, model, device=args.gpu),
                   trigger=(args.validation_interval, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    if args.early_stopping:
        # Take a best snapshot
        record_trigger = training.triggers.MaxValueTrigger(
            'validation/main/loss', (args.validation_interval, 'epoch'))
        trainer.extend(
            extensions.snapshot_object(model, 'best_model.npz'),
            trigger=record_trigger)

    trainer.run()

    if args.early_stopping:
        chainer.serializers.load_npz(
            os.path.join(args.out, 'best_model.npz'), model)
    else:
        chainer.serializers.save_npz(
            os.path.join(args.out, 'best_model.npz'), model)

    print('Running test...')
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        _, accuracy = model.evaluate(idx_test)
    print('Test accuracy = %f' % accuracy)



if __name__ == '__main__':
    main()