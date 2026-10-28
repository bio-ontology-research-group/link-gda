"""Four checkpoint choices on one fixed-budget training trajectory."""
import json
import math
from pathlib import Path
import torch
from pykeen.training.callbacks import TrainingCallback


def head_key(metrics):
    values = tuple(float(metrics[k]) for k in ('hits@10', 'hits@1', 'mrr', 'mr'))
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite validation metric')
    return (*values[:3], -values[3])


class FixedBudgetHeadCallback(TrainingCallback):
    """No early stopping; evaluate() supplies raw/calibrated BMA metrics.

    PyKEEN 1.11.1 training_loop supplies one-based completed epochs. Checkpoints
    contain plain model.state_dict(), and the curve records selected epochs.
    """
    def __init__(self, model, evaluate, checkpoint_prefix, curve_path, frequency=20, resume=False):
        super().__init__()
        self.selection_model = model
        self.evaluate = evaluate
        self.prefix = Path(checkpoint_prefix)
        self.curve_path = Path(curve_path)
        if frequency < 1:
            raise ValueError('frequency must be positive')
        self.frequency = frequency
        self.best = {}
        self.curve = []
        self.checkpoint_paths = {f'{arm}_{criterion}': Path(f'{self.prefix}_{arm}_{criterion}.pt')
                                 for arm in ('raw', 'calibrated') for criterion in ('mr', 'head')}
        for path in [self.curve_path, *self.checkpoint_paths.values()]:
            if path.exists() and not resume:
                raise FileExistsError(f'Refusing existing trajectory artifact: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_files(self):
        return [p for p in [self.curve_path, *self.checkpoint_paths.values()] if p.is_file()]

    def state_dict(self):
        return {'frequency': self.frequency, 'best': self.best, 'curve': self.curve}

    def load_state_dict(self, state):
        if state['frequency'] != self.frequency:
            raise ValueError('Validation frequency changed')
        self.best = state['best']
        self.curve = state['curve']
        for key, value in self.best.items():
            value['checkpoint'] = str(self.checkpoint_paths[key])

    def post_epoch(self, epoch, epoch_loss, **kwargs):
        if epoch % self.frequency == 0:
            self.evaluate_epoch(epoch, epoch_loss)

    def evaluate_epoch(self, epoch, epoch_loss=None):
        if self.curve and epoch <= self.curve[-1]['epoch']:
            raise ValueError('Validation epochs must increase strictly')
        model = self.selection_model
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                result = self.evaluate()
            metrics = {arm: {k: float(v) for k, v in (result[arm][0] if isinstance(result[arm], (tuple, list)) else result[arm]).items()}
                       for arm in ('raw', 'calibrated')}
            for arm, values in metrics.items():
                head_key(values)  # fail before writing any checkpoints
            for arm, values in metrics.items():
                for criterion in ('mr', 'head'):
                    name = f'{arm}_{criterion}'
                    key = (-values['mr'],) if criterion == 'mr' else head_key(values)
                    if name not in self.best or key > tuple(self.best[name]['key']):
                        path = self.checkpoint_paths[name]
                        temporary = path.with_suffix('.pt.tmp')
                        torch.save(model.state_dict(), temporary)
                        temporary.replace(path)
                        self.best[name] = {'epoch': int(epoch), 'key': list(key), 'metrics': values,
                                           'checkpoint': str(path)}
            self.curve.append({'epoch': int(epoch), 'loss': None if epoch_loss is None else float(epoch_loss),
                               'metrics': metrics})
            temporary = self.curve_path.with_suffix(self.curve_path.suffix + '.tmp')
            temporary.write_text(json.dumps({'schema_version': 1, 'frequency': self.frequency,
                                            'best': self.best, 'curve': self.curve}, indent=2, allow_nan=False) + '\n')
            temporary.replace(self.curve_path)
        finally:
            if hasattr(model, '_free_graph_and_cache'):
                model._free_graph_and_cache()
            model.train(was_training)
