"""
No@
Oct 21st, 2024
"""

from mmengine.registry import HOOKS
from mmengine.hooks import EarlyStoppingHook
from mmengine.dist import is_main_process


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
        # if self.wait_count == 0:
        #     runner.visualizer._vis_backends['WandbVisBackend']._wandb.summary['val/combined_score'] = metrics['combined_score']
        #     runner.visualizer._vis_backends['WandbVisBackend']._wandb.summary['val/SIC.r2'] = metrics['SIC']['r2']
        #     runner.visualizer._vis_backends['WandbVisBackend']._wandb.summary['val/SOD.f1'] = metrics['SOD']['f1']
        #     runner.visualizer._vis_backends['WandbVisBackend']._wandb.summary['val/FLOE.f1'] = metrics['FLOE']['f1']

