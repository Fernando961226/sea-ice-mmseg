# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from mmengine.registry import HOOKS
from mmengine.hooks import RuntimeInfoHook

def _is_scalar(value: Any) -> bool:
    """Determine the value is a scalar type value.

    Args:
        value (Any): value of log.

    Returns:
        bool: whether the value is a scalar type value.
    """
    if isinstance(value, np.ndarray):
        return value.size == 1
    elif isinstance(value, (int, float, np.number)):
        return True
    elif isinstance(value, torch.Tensor):
        return value.numel() == 1
    return False


@HOOKS.register_module()
class AI4arcticRuntimeInfoHook(RuntimeInfoHook):
    """A hook that updates runtime information into message hub.

    E.g. ``epoch``, ``iter``, ``max_epochs``, and ``max_iters`` for the
    training state. Components that cannot access the runner can get runtime
    information through the message hub.
    """

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        if metrics is not None:
            for key, value in metrics.items():
                if value is None: continue
                if not _is_scalar(value):
                    if key == 'SIC':
                        value = value['r2']
                        key += '.r2'
                    elif key in ['SOD', 'FLOE']:
                        value = value['f1']
                        key += '.f1'
                if _is_scalar(value):
                    runner.message_hub.update_scalar(f'val/{key}', value)
                else:
                    runner.message_hub.update_info(f'val/{key}', value)
            if value is not None:       # we met some condition in the loop
                runner.message_hub.update_scalar(f'val/train_iter', runner._train_loop._iter)


    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        if metrics is not None:
            for key, value in metrics.items():
                if value is None: continue
                if not _is_scalar(value):
                    if key == 'SIC':
                        value = value['r2']
                        key += '.r1'
                    elif key in ['SOD', 'FLOE']:
                        value = value['f1']
                        key += '.f1'
                if _is_scalar(value):
                    runner.message_hub.update_scalar(f'test/{key}', value)
                else:
                    runner.message_hub.update_info(f'test/{key}', value)
