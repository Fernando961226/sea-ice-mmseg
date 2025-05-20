"""
No@
Oct 21st, 2024
"""

from mmengine.registry import HOOKS
from mmengine.hooks import EarlyStoppingHook
from mmengine.dist import is_main_process, broadcast_object_list

@HOOKS.register_module()  
class EarlyStoppingHookMain(EarlyStoppingHook):

    def after_val_epoch(self, runner, metrics):
        """Decide whether to stop the training process.
        Added by No@:
            Check only if is_main_process() is True

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """

        if not is_main_process(): return
        super().after_val_epoch(runner, metrics)
        if runner.train_loop.stop_training:
            # Broadcast the stop flag
            object_list = [runner.train_loop.stop_training]
            broadcast_object_list(object_list, src=0)
            stop_flag = object_list[0]
            # Update runner in all processes based on the stop_flag
            runner.train_loop.stop_training = stop_flag