import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.sklearns import R2Score


class LinearBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = self.generate_layers(*args, **kwargs)
        self.out_sz = kwargs['out_sz']
            
    def generate_layers(self, in_sz, layers, out_sz, ps, use_bn, bn_final):
        if ps is None: ps = [0]*len(layers) 
        else: ps = ps*len(layers)
        sizes = self.get_sizes(in_sz, layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += self.bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        block = nn.Sequential(*layers)
        return block
        
    def get_sizes(self, in_sz, layers, out_sz):
        return [in_sz] + layers + [out_sz]
    
    def bn_drop_lin(self, n_in:int, n_out:int, bn:bool=True, p:float=0., actn:nn.Module=None):
        "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        return layers
    
    def forward(self, x):
        x = self.block(x)
        return x

    
class FiLMGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gamma = LinearBlock(*args, **kwargs)
        self.beta = LinearBlock(*args, **kwargs)
    
    def forward(self, x):
        gamma = self.gamma(x)
        beta = self.beta(x)
        return gamma, beta

    
class FiLMNetwork(pl.LightningModule):
    def __init__(self, inputs_sz, conds_sz, learning_rate=1e-2, metric=R2Score()):
        super().__init__()
        self.save_hyperparameters()
        self.metric = metric
        self.inputs_emb = LinearBlock(in_sz=inputs_sz, layers=[512,256,128,64], out_sz=32, ps=None, use_bn=True, bn_final=True)
        self.conds_emb = LinearBlock(in_sz=conds_sz, layers=[], out_sz=32, ps=None, use_bn=True, bn_final=False)
        self.film_1 = FiLMGenerator(in_sz=32, layers=[], out_sz=32, ps=None, use_bn=False, bn_final=False)
        self.block_1 = LinearBlock(in_sz=32, layers=[16], out_sz=16, ps=None, use_bn=True, bn_final=True)
        self.film_2 = FiLMGenerator(in_sz=32, layers=[], out_sz=16, ps=None, use_bn=False, bn_final=False)
        self.block_2 = LinearBlock(in_sz=16, layers=[8], out_sz=1, ps=None, use_bn=True, bn_final=False)
    
    def forward(self, conds):
        return self.conds_emb(conds)

    def training_step(self, batch, batch_idx):
        inputs, conds, y = batch
        input_emb = self.inputs_emb(inputs)
        conds_emb = self.conds_emb(conds)
        gamma_1, beta_1 = self.film_1(conds_emb)
        x = input_emb * gamma_1 + beta_1
        x = self.block_1(x)
        gamma_2, beta_2 = self.film_2(conds_emb)
        x = x * gamma_2 + beta_2
        y_hat = self.block_2(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result
    
    def validation_step(self, batch, batch_idx):
        inputs, conds, y = batch
        input_emb = self.inputs_emb(inputs)
        conds_emb = self.conds_emb(conds)
        gamma_1, beta_1 = self.film_1(conds_emb)
        x = input_emb * gamma_1 + beta_1
        x = self.block_1(x)
        gamma_2, beta_2 = self.film_2(conds_emb)
        x = x * gamma_2 + beta_2
        y_hat = self.block_2(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, on_step=True)
        result.log('val_r2', self.metric(y_hat, y), on_step=True)
        return result

    def test_step(self, batch, batch_idx):
        inputs, conds, y = batch
        input_emb = self.inputs_emb(inputs)
        conds_emb = self.conds_emb(conds)
        gamma_1, beta_1 = self.film_1(conds_emb)
        x = input_emb * gamma_1 + beta_1
        x = self.block_1(x)
        gamma_2, beta_2 = self.film_2(conds_emb)
        x = x * gamma_2 + beta_2
        y_hat = self.block_2(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss, on_step=True)
        result.log('test_r2', self.metric(y_hat, y), on_step=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    