"""独立停止头实验：冻结解码特征、同目标训练，不改变模型/Trainer源码。"""
from copy import deepcopy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


HEAD_KEYS={'decoder.stop_head.weight','decoder.stop_head.bias'}


def backbone_digest(model_or_state):
    state=model_or_state.state_dict() if isinstance(model_or_state,nn.Module) else model_or_state
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if name in HEAD_KEYS:continue
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str((value.dtype,tuple(value.shape))).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_stop_features(model,batches,mode,*,progress=None):
    """捕获stop_head只读输入；hook不参与计算，失败时同样清除。"""
    if mode not in {'guided','free'} or next(model.parameters()).device.type!='cpu':raise ValueError('仅允许CPU guided/free')
    head=model.decoder.stop_head
    hidden=[];parts={key:[] for key in ('features','logits','points','lengths','targets','masks','goals')}
    was_training=model.training;model.eval()
    hook=head.register_forward_pre_hook(lambda module,args:hidden.append(args[0].detach().cpu().clone()))
    try:
        with torch.no_grad():
            for index,batch in enumerate(batches):
                hidden.clear();bev,goal,state,target,mask=batch[:5]
                output=model.forward_with_stop(bev,goal,state,**({'teacher_points':target,'teacher_forcing_ratio':1.} if mode=='guided' else {}))
                if len(hidden)!=model.max_horizon:raise ValueError('特征hook次数不符')
                features=torch.stack(hidden,dim=1)
                logits=head(features).squeeze(-1)
                if not torch.isfinite(features).all() or not torch.allclose(logits,output.stop_logits,atol=1e-6,rtol=0):raise ValueError('缓存特征/logit不等价')
                values={'features':features,'logits':output.stop_logits,'points':output.points,'lengths':mask.sum(1).long(),
                        'targets':target,'masks':mask,'goals':goal}
                for key,value in values.items():parts[key].append(value.detach().cpu())
                if progress:progress(index+1)
        if not parts['features']:raise ValueError('缓存输入为空')
        return {key:torch.cat(values) for key,values in parts.items()}
    finally:
        hook.remove();model.train(was_training)


def cumulative_stop_loss(logits,lengths):
    if (logits.ndim!=2 or lengths.shape!=(len(logits),) or not len(logits)
            or not torch.isfinite(logits).all() or not torch.isfinite(lengths).all()
            or not torch.equal(lengths,lengths.long()) or torch.any(lengths<=0) or torch.any(lengths>logits.shape[1])):
        raise ValueError('停止logit/有效长度不合法')
    targets=(torch.arange(logits.shape[1],device=logits.device)[None]>=lengths[:,None]-1).to(logits.dtype)
    return .2*F.binary_cross_entropy_with_logits(logits,targets)


def fit_stop_head(initial_head,cache,*,epochs=20,learning_rate=2e-4,batch_size=8,seed=20261020,on_epoch=None):
    features,lengths=cache['features'],cache['lengths']
    if (features.ndim!=3 or features.shape[-1]!=initial_head.in_features or not len(features)
            or lengths.shape!=(len(features),) or not torch.isfinite(features).all()
            or not isinstance(epochs,int) or epochs<1 or not isinstance(batch_size,int) or batch_size<1
            or not math.isfinite(learning_rate) or learning_rate<=0):raise ValueError('训练特征或预算不合法')
    # 只优化独立复制的head；源模型的requires_grad与参数均不改写。
    head=deepcopy(initial_head).cpu();head.train()
    for parameter in head.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.Adam(head.parameters(),lr=learning_rate);history=[]
    for epoch in range(epochs):
        order=torch.randperm(len(features),generator=torch.Generator().manual_seed(seed+epoch));total=0.
        for begin in range(0,len(order),batch_size):
            indices=order[begin:begin+batch_size];optimizer.zero_grad(set_to_none=True)
            loss=cumulative_stop_loss(head(features[indices]).squeeze(-1),lengths[indices])
            if not torch.isfinite(loss):raise ValueError('停止头loss非有限')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(indices)
        history.append(total/len(features))
        if on_epoch:on_epoch(epoch+1,head,history.copy())
    return head,history


def make_head_checkpoint(parent,head,*,epoch,metadata):
    checkpoint=deepcopy(parent)
    for name,value in head.state_dict().items():checkpoint['model_state']['decoder.stop_head.'+name]=value.detach().cpu().clone()
    if backbone_digest(checkpoint['model_state'])!=backbone_digest(parent['model_state']):raise ValueError('冻结主干变化')
    checkpoint.update({'epoch':epoch-1,'best_epoch':None,'best_val_loss':None,'optimizer_state':None,'resumable':False,
                       'history':{'stop_head_loss':metadata.get('loss_history',[])},
                       'head_adaptation':{**metadata,'source_epoch':parent.get('epoch'),'adaptation_epoch':epoch},
                       'stop_calibration':{'policy':'inherited_parent_threshold_not_recalibrated','threshold':parent['model_config']['stop_threshold']}})
    return checkpoint
