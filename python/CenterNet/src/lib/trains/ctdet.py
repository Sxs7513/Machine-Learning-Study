from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from .base_trainer import BaseTrainer


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

    # hm => [N, C(类别数), 128, 128] 
    # wh => [N, 2, 128, 128] 
    def forward(self, output, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output["hm"] = _sigmoid(output['hm'])

            # TODO eval_oracle

            hm_loss += self.crit(output["hm"], batch["hm"]) / opt.num_stacks
            
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    pass
                elif opt.cat_spec_wh:
                    pass
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                            batch['ind'], batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
      # TODO   
      return
