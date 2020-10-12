import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import r2_score


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
        self.out_sz = kwargs['out_sz']
    
    def forward(self, x):
        gamma = self.gamma(x)
        beta = self.beta(x)
        return gamma, beta

    
class FiLMNetwork(pl.LightningModule):
    """Single FiLM generator."""
    def __init__(self, inputs_sz, conds_sz, learning_rate=1e-3, batch_size=2048, metric=r2_score):
        super().__init__()
        self.save_hyperparameters()
        self.metric = metric
        self.inputs_emb = LinearBlock(in_sz=inputs_sz, layers=[512,256,128,64], out_sz=32, ps=None, use_bn=True, bn_final=True)
        self.conds_emb = LinearBlock(in_sz=conds_sz, layers=[], out_sz=32, ps=None, use_bn=True, bn_final=False)
        self.film_gen = FiLMGenerator(in_sz=self.conds_emb.out_sz, layers=[], out_sz=32, ps=None, use_bn=False, bn_final=False)
        self.block_1 = LinearBlock(in_sz=self.film_gen.out_sz, layers=[16], out_sz=32, ps=None, use_bn=True, bn_final=True)
        self.block_2 = LinearBlock(in_sz=self.film_gen.out_sz, layers=[8], out_sz=1, ps=None, use_bn=True, bn_final=False)
    
    def forward(self, inputs, conds_a, conds_b):
        inputs_emb = self.inputs_emb(inputs)
        conds_a_emb = self.conds_emb(conds_a)
        conds_b_emb = self.conds_emb(conds_b)
        gamma_a, beta_a = self.film_gen(conds_a_emb)
        gamma_b, beta_b = self.film_gen(conds_b_emb)
        x = inputs_emb * gamma_a + beta_a
        x = self.block_1(x)
        x = x * gamma_b + beta_b
        y_hat = self.block_2(x)
        y_hat = torch.clamp(y_hat, min=0)
        return inputs_emb, conds_a_emb, conds_b_emb, y_hat

    def training_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    
class MultiFiLMNetwork(pl.LightningModule):
    """Dual FiLM generators."""
    def __init__(self, inputs_sz, conds_sz, learning_rate=1e-3, batch_size=2048, metric=r2_score):
        super().__init__()
        self.save_hyperparameters()
        self.metric = metric
        self.inputs_emb = LinearBlock(in_sz=inputs_sz, layers=[512,256,128,64], out_sz=32, ps=None, use_bn=True, bn_final=True)
        self.conds_emb = LinearBlock(in_sz=conds_sz, layers=[], out_sz=32, ps=None, use_bn=True, bn_final=False)
        self.film_1 = FiLMGenerator(in_sz=self.conds_emb.out_sz, layers=[], out_sz=32, ps=None, use_bn=False, bn_final=False)
        self.block_1 = LinearBlock(in_sz=self.film_1.out_sz, layers=[16], out_sz=16, ps=None, use_bn=True, bn_final=True)
        self.film_2 = FiLMGenerator(in_sz=self.conds_emb.out_sz, layers=[], out_sz=16, ps=None, use_bn=False, bn_final=False)
        self.block_2 = LinearBlock(in_sz=self.film_2.out_sz, layers=[8], out_sz=1, ps=None, use_bn=True, bn_final=False)
    
    def forward(self, inputs, conds_a, conds_b):
        inputs_emb = self.inputs_emb(inputs)
        conds_a_emb = self.conds_emb(conds_a)
        conds_b_emb = self.conds_emb(conds_b)
        gamma_1, beta_1 = self.film_1(conds_a_emb)
        x = inputs_emb * gamma_1 + beta_1
        x = self.block_1(x)
        gamma_2, beta_2 = self.film_2(conds_b_emb)
        x = x * gamma_2 + beta_2
        y_hat = self.block_2(x)
        y_hat = torch.clamp(y_hat, min=0)
        return inputs_emb, conds_a_emb, conds_b_emb, y_hat

    def training_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class ScaleNetwork(pl.LightningModule):
    """Conditional scaling network."""
    def __init__(self, inputs_sz, conds_sz, learning_rate=1e-3, batch_size=2048, metric=r2_score):
        super().__init__()
        self.save_hyperparameters()
        self.metric = metric
        self.inputs_emb = LinearBlock(in_sz=inputs_sz, layers=[512,256,128,64], out_sz=32, ps=None, use_bn=True, bn_final=True)
        self.conds_emb = LinearBlock(in_sz=conds_sz, layers=[], out_sz=32, ps=None, use_bn=True, bn_final=False)
        self.film_gen = FiLMGenerator(in_sz=self.conds_emb.out_sz, layers=[], out_sz=32, ps=None, use_bn=False, bn_final=False)
        self.block_1 = LinearBlock(in_sz=self.film_gen.out_sz, layers=[16], out_sz=32, ps=None, use_bn=True, bn_final=True)
        self.block_2 = LinearBlock(in_sz=self.film_gen.out_sz, layers=[8], out_sz=1, ps=None, use_bn=True, bn_final=False)
    
    def forward(self, inputs, conds_a, conds_b):
        inputs_emb = self.inputs_emb(inputs)
        conds_a_emb = self.conds_emb(conds_a)
        conds_b_emb = self.conds_emb(conds_b)
        gamma_a, beta_a = self.film_gen(conds_a_emb)
        gamma_b, beta_b = self.film_gen(conds_b_emb)
        x = inputs_emb * gamma_a + beta_a
        x = self.block_1(x)
        x = x * gamma_b + beta_b
        y_hat = self.block_2(x)
        y_hat = torch.clamp(y_hat, min=0)
        return inputs_emb, conds_a_emb, conds_b_emb, y_hat

    def training_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_a_emb, conds_b_emb, y_hat = self.forward(inputs, conds, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class ConcatNetwork(pl.LightningModule):
    """Conditional biasing network."""
    def __init__(self, inputs_sz, conds_sz, learning_rate=1e-3, batch_size=2048, metric=r2_score):
        super().__init__()
        self.save_hyperparameters()
        self.metric = metric
        self.inputs_emb = LinearBlock(in_sz=inputs_sz, layers=[512,256,128,64], out_sz=32, ps=None, use_bn=True, bn_final=True)
        self.conds_emb = LinearBlock(in_sz=conds_sz, layers=[], out_sz=32, ps=None, use_bn=True, bn_final=False)
        self.block_1 = LinearBlock(in_sz=self.inputs_emb.out_sz + self.conds_emb.out_sz, layers=[16], out_sz=16, ps=None, use_bn=True, bn_final=True)
        self.block_2 = LinearBlock(in_sz=self.block_1.out_sz + self.conds_emb.out_sz, layers=[8], out_sz=1, ps=None, use_bn=True, bn_final=False)
    
    def forward(self, inputs, conds):
        inputs_emb = self.inputs_emb(inputs)
        conds_emb = self.conds_emb(conds)
        x = torch.cat([inputs_emb, conds_emb], dim=1)
        x = self.block_1(x)
        x = torch.cat([x, conds_emb], dim=1)
        y_hat = self.block_2(x)
        y_hat = torch.clamp(y_hat, min=0)
        return inputs_emb, conds_emb, y_hat
    
    def training_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_emb, y_hat = self.forward(inputs, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_emb, y_hat = self.forward(inputs, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, conds, y = batch
        inputs_emb, conds_emb, y_hat = self.forward(inputs, conds)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_r2', self.metric(y.detach().cpu(), y_hat.detach().cpu()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)