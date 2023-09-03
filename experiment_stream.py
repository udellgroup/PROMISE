import random 
from utils import *
import timeit
from models.least_squares_stream import LeastSquaresStream
from models.logistic_stream import LogisticStream
from opts.sgd_stream import SGDStream
from opts.saga_stream import SAGAStream
from opts.sketchysgd_stream import SketchySGDStream
from opts.sketchysaga_stream import SketchySAGAStream

class ExperimentStream():
    def __init__(self, 
                data_obj, 
                model_type, 
                model_params, 
                opt_name, 
                opt_params,
                time_budget = None,
                verbose = False):
        self.verbose = verbose

        # Instantiate the model
        self.model_type = model_type
        self.model_params = model_params
        self.model = self.create_model(data_obj)

        # Instantiate the optimizer
        self.opt_name = opt_name
        self.opt_params = opt_params

        # Set the stopping criteria
        self.check_time = time_budget is not None
        self.time_budget = time_budget

    # Return a model of the desired type with the requested data
    def create_model(self, data_obj):
        # Create the model
        if self.model_type == 'logistic':
            model = LogisticStream(data_obj, **self.model_params)
        elif self.model_type == 'least_squares':
            model = LeastSquaresStream(data_obj, **self.model_params)
        else:
            raise RuntimeError(f"We do not currently support the following model: {self.model_type}.")
        return model

    # Return an optimizer that we can use to train the model
    def create_opt(self):
        if self.opt_name == 'sgd':
            opt = SGDStream(self.model, **self.opt_params)
        elif self.opt_name == 'saga':
            opt = SAGAStream(self.model, **self.opt_params)
        elif self.opt_name == 'sketchysgd':
            opt = SketchySGDStream(self.model, **self.opt_params)
        elif self.opt_name == 'sketchysaga':
            opt = SketchySAGAStream(self.model, **self.opt_params)
        else:
            raise RuntimeError(f"We do not currently support the following optimizer in the streaming setting: {self.opt_name}.")
        return opt

    # If update frequency is given in epochs, convert to minibatches
    def preprocess_opt_params(self, bg):
        n_batches = int(np.ceil(self.model.ntr/bg))
        if 'update_freq' in self.opt_params.keys():
            for freq_type, freq_pair in self.opt_params['update_freq'].items():
                if freq_pair[1] == 'epochs':
                    self.opt_params['update_freq'][freq_type] = freq_pair[0] * n_batches
                elif freq_pair[1] == 'minibatches':
                    self.opt_params['update_freq'][freq_type] = freq_pair[0]
                else:
                    raise RuntimeError(f"We do not currently support the following update frequency type: {freq_pair[1]}.")

    # Run the experiment
    def run(self, max_epochs, bg):
        self.preprocess_opt_params(bg)
        opt = self.create_opt()
        if hasattr(opt, 'update_freq'): print(opt.update_freq)

        results = {'times': [], 'data_upd_times': [], 'test_loss': [], 'test_acc': [], 'eta': []}

        # Training loop
        for i in range(max_epochs):
            data_upd_time_total = 0
            epoch_start = timeit.default_timer()

            batches = minibatch_indices(self.model.ntr, bg)
            # Loop through every minibatch
            for batch in batches:
                data_upd_time = opt.step(batch)
                data_upd_time_total += data_upd_time
            
            epoch_end = timeit.default_timer()
            results['times'].append(epoch_end - epoch_start)
            results['data_upd_times'].append(data_upd_time_total)

            # Get the results so far
            losses = self.model.get_losses()
            results['test_loss'].append(losses['test_loss'])

            accs = self.model.get_acc()
            results['test_acc'].append(accs['test_acc'])

            results['eta'].append(opt.eta)

            if self.verbose:
                print(f"Test loss at epoch {i}: {losses['test_loss']}")
                print(f"Test acc. at epoch {i}: {accs['test_acc']}")

            # If time budget is met, stop training
            if self.check_time and sum(results['times']) > self.time_budget:
                break

        return results

