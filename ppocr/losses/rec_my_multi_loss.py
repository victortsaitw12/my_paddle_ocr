# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss


class SoftCrossEntropyLoss(nn.Layer):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax: 
            log_prob = F.log_softmax(input, axis=-1)
        else: 
            log_prob = paddle.log(input)
        loss = -(target * log_prob).sum(axis=-1)
        if self.reduction == "mean": 
            return loss.mean()
        
        elif self.reduction == "sum":
            return loss.sum()
        
        else: return loss

class MyMultiLoss(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.one_hot = kwargs.get('one_hot', False)
        ignore_index = kwargs.get('ignore_index', -1)
        if self.one_hot:
            self.loss_func = SoftCrossEntropyLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss(
                    reduction='mean', ignore_index=ignore_index)

    def _ce_loss(self, out, tgt, weight=1.0):
     
        iter_size = out.shape[0] // tgt.shape[0]
        
        if iter_size > 1:
            tgt = paddle.concat([tgt] * iter_size, 0)
            # tgt = tgt.tile(iter_size, 1, 1)
      
        flat_out = out.reshape([-1, out.shape[-1]])
        if self.one_hot:
            flat_gt = tgt.reshape([-1, tgt.shape[-1]])
        else:
            flat_gt = tgt.reshape([-1])
     
        return self.loss_func(flat_out, flat_gt) * weight
            
    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res    
        return paddle.concat(all_res, axis=0)
      
    
    def forward(self, predicts, batch):
        # print('predicts:', predicts.shape)
        # print('batch[0]:', batch[0])
        # print('batch[1]:', batch[1])
        # print('batch[2]:', batch[2])

        self.total_loss = {}

        # if isinstance(predicts, (tuple, list)):
        if isinstance(predicts, (dict)):
            outputs = [self._merge_list(o) for o in predicts.values()]
            loss = sum([self._ce_loss(o, batch[1]) for o in outputs])
        else:
            loss = self._ce_loss(predicts, batch[1])
        self.total_loss['loss'] = loss
        return self.total_loss
