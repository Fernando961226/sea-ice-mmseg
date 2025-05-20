"""
No@
"""
import os.path as osp
from typing import Dict, Optional


from mmengine.fileio import dump
from mmengine.registry import HOOKS
from mmengine.hooks import LoggerHook

@HOOKS.register_module()
class AI4arcticLoggerHook(LoggerHook):
    """
    Modified by No@:
        log test metrics to visualizer
    """
    
    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.test_dataloader), 'test', with_non_scalar=True)
        runner.logger.info(log_str)
        dump(
            self._process_tags(tag),
            osp.join(runner.log_dir, self.json_log_path))  # type: ignore
        
        runner.visualizer.add_scalars(
            tag, step=runner.iter, file_path=self.json_log_path)