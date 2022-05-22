import torch
import torch.nn as nn
import random
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torchcde

class F(torch.nn.Module):
    def __init__(self,hidden_channels,input_channels):
        super(F, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(self.hidden_channels,128)
        self.linear2 = torch.nn.Linear(self.hidden_channels, self.hidden_channels * self.input_channels)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()

        return z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

class Encoder1(nn.Module):
    def __init__(self, device, rnn_type, input_size, hidden_size=64,
                 num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.device = device
        self.rnn_type = rnn_type
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hidden_dim=64
        self.number_nodes=307
        self.feature_dim=4

        self.initial = nn.Linear(self.number_nodes * self.feature_dim + 1, self.hidden_dim)


    def forward(self, x):

        # x = [seq_len, batch_size, input_size=number_nodes*feature_dim]
        # h_0 = [layers * num_directions, batch_size, hidden_size]
        x=x[:4,:,:]
        seq_len, batch_size, input_size=x.shape
        x= x.permute(1,0,2)#[batch_size, seq_len, input_size=number_nodes*feature_dim]
        # inputs = inputs.permute(0, 2, 1, 3)  # (input_window, batch_size, num_nodes, input_dim)
        # inputs = inputs.reshape(batch_size * num_nodes, input_window, feature_dim).to(self.device)
        t = torch.linspace(0, 3, 4).to(self.device)
        #t = torch.tensor([0,1,2,3]).float().to(self.device)
        tt=torch.tensor([3]).float().to(self.device)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch_size , seq_len, 1).to(self.device)
        inputs = torch.cat([t_, x], dim=2)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inputs).to(self.device)
        X = torchcde.CubicSpline(coeffs).to(self.device)
        func = F(hidden_channels=self.hidden_dim, input_channels=input_size+ 1).to(self.device)
        z0 = self.initial(inputs[:, 0, :]).to(self.device)
        y = torchcde.cdeint(X=X, func=func, z0=z0, t=tt).to(self.device)  # [batch_size ,tt_dim, hidden_dim]
        hn = y.permute(1,0,2)
        cn = torch.zeros(hn.shape)
        return hn, cn

class Encoder2(nn.Module):
    def __init__(self, device, rnn_type, input_size, hidden_size=64,
                 num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.device = device
        self.rnn_type = rnn_type
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hidden_dim=64
        self.number_nodes=307
        self.feature_dim=4

        self.initial = nn.Linear(self.number_nodes * self.feature_dim + 1, self.hidden_dim)


    def forward(self, x):

        # x = [seq_len, batch_size, input_size=number_nodes*feature_dim]
        # h_0 = [layers * num_directions, batch_size, hidden_size]
        x=x[4:8,:,:]
        seq_len, batch_size, input_size=x.shape
        x= x.permute(1,0,2)#[batch_size, seq_len, input_size=number_nodes*feature_dim]
        # inputs = inputs.permute(0, 2, 1, 3)  # (input_window, batch_size, num_nodes, input_dim)
        # inputs = inputs.reshape(batch_size * num_nodes, input_window, feature_dim).to(self.device)
        t = torch.linspace(0, 3, 4).to(self.device)
        #t = torch.tensor([0,1,2]).float().to(self.device)
        tt=torch.tensor([3]).float().to(self.device)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch_size , seq_len, 1).to(self.device)
        inputs = torch.cat([t_, x], dim=2)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inputs).to(self.device)
        X = torchcde.CubicSpline(coeffs).to(self.device)
        func = F(hidden_channels=self.hidden_dim, input_channels=input_size+ 1).to(self.device)
        z0 = self.initial(inputs[:, 0, :]).to(self.device)
        y = torchcde.cdeint(X=X, func=func, z0=z0, t=tt).to(self.device)  # [batch_size ,tt_dim, hidden_dim]
        hn = y.permute(1,0,2)
        cn = torch.zeros(hn.shape)
        return hn, cn

class Encoder3(nn.Module):
    def __init__(self, device, rnn_type, input_size, hidden_size=64,
                 num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.device = device
        self.rnn_type = rnn_type
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hidden_dim=64
        self.number_nodes=307
        self.feature_dim=4

        self.initial = nn.Linear(self.number_nodes * self.feature_dim + 1, self.hidden_dim)


    def forward(self, x):

        # x = [seq_len, batch_size, input_size=number_nodes*feature_dim]
        # h_0 = [layers * num_directions, batch_size, hidden_size]
        x=x[-4:,:,:]
        seq_len, batch_size, input_size=x.shape
        x= x.permute(1,0,2)#[batch_size, seq_len, input_size=number_nodes*feature_dim]
        # inputs = inputs.permute(0, 2, 1, 3)  # (input_window, batch_size, num_nodes, input_dim)
        # inputs = inputs.reshape(batch_size * num_nodes, input_window, feature_dim).to(self.device)
        t = torch.linspace(0, 3, 4).to(self.device)
        #t = torch.tensor([1,2,3]).float().to(self.device)
        tt=torch.tensor([3]).float().to(self.device)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch_size , seq_len, 1).to(self.device)
        inputs = torch.cat([t_, x], dim=2)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inputs).to(self.device)
        X = torchcde.CubicSpline(coeffs).to(self.device)
        func = F(hidden_channels=self.hidden_dim, input_channels=input_size+ 1).to(self.device)
        z0 = self.initial(inputs[:, 0, :]).to(self.device)
        y = torchcde.cdeint(X=X, func=func, z0=z0, t=tt).to(self.device)  # [batch_size ,tt_dim, hidden_dim]
        hn = y.permute(1,0,2)
        cn = torch.zeros(hn.shape)
        return hn, cn


class Decoder(nn.Module):
    def __init__(self, device, rnn_type, input_size, hidden_size=64,
                 num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.device = device
        self.rnn_type = rnn_type
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))
        self.fc = nn.Linear(hidden_size * self.num_directions, input_size)

    def forward(self, x, hn, cn):
        # x = [batch_size, input_size]
        # hn, cn = [layers * num_directions, batch_size, hidden_size]
        x = x.unsqueeze(0)
        # x = [seq_len = 1, batch_size, input_size]
        if self.rnn_type == 'LSTM':
            out, (hn, cn) = self.rnn(x, (hn, cn))
        else:
            out, hn = self.rnn(x, hn)
            cn = torch.zeros(hn.shape)
        # out = [seq_len = 1, batch_size, hidden_size * num_directions]
        # hn = [layers * num_directions, batch_size, hidden_size]
        out = self.fc(out.squeeze(0))
        # out = [batch_size, input_size]
        return out, hn, cn


class Seq2Seq(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
 
#初始化
        self.fuse_weight_1.data.fill_(0.33)
        self.fuse_weight_2.data.fill_(0.33)
        self.fuse_weight_3.data.fill_(0.33)



        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.rnn_type = config.get('rnn_type', 'GRU')
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        self.bidirectional = config.get('bidirectional', False)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)
        self.encoder1 = Encoder1(self.device, self.rnn_type, self.num_nodes * self.feature_dim,
                               self.hidden_size, self.num_layers, self.dropout, self.bidirectional)
        self.encoder2 = Encoder2(self.device, self.rnn_type, self.num_nodes * self.feature_dim,
                               self.hidden_size, self.num_layers, self.dropout, self.bidirectional)
        self.encoder3 = Encoder3(self.device, self.rnn_type, self.num_nodes * self.feature_dim,
                               self.hidden_size, self.num_layers, self.dropout, self.bidirectional)
        self.decoder = Decoder(self.device, self.rnn_type, self.num_nodes * self.output_dim,
                               self.hidden_size, self.num_layers, self.dropout, self.bidirectional)
        self._logger.info('You select rnn_type {} in Seq2Seq!'.format(self.rnn_type))

    def forward(self, batch):
        src = batch['X']  # [batch_size, input_window, num_nodes, feature_dim]
        target = batch['y']  # [batch_size, output_window, num_nodes, feature_dim]
        src = src.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, feature_dim]
        target = target.permute(1, 0, 2, 3)  # [output_window, batch_size, num_nodes, feature_dim]

        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)
        target = target[..., :self.output_dim].contiguous().reshape(
            self.output_window, batch_size, self.num_nodes * self.output_dim)
        # src = [self.input_window, batch_size, self.num_nodes * self.feature_dim]
        # target = [self.output_window, batch_size, self.num_nodes * self.output_dim]

        encoder_hn1, encoder_cn = self.encoder1(src)
        encoder_hn2, encoder_cn = self.encoder2(src)
        encoder_hn3, encoder_cn = self.encoder3(src)
        encoder_hn1=encoder_hn1.to(self.device)
        encoder_hn2=encoder_hn2.to(self.device)
        encoder_hn3=encoder_hn3.to(self.device)
        encoder_cn=encoder_cn.to(self.device)
        
        self.fuse_weight_1=self.fuse_weight_1.to(self.device)
        self.fuse_weight_2=self.fuse_weight_2.to(self.device)
        self.fuse_weight_3=self.fuse_weight_3.to(self.device)
        encoder_hn=self.fuse_weight_1*encoder_hn1+self.fuse_weight_2*encoder_hn2+self.fuse_weight_3*encoder_hn3
        
        
        decoder_hn = encoder_hn
        decoder_cn = encoder_cn
        # encoder_hidden_state = [layers * num_directions, batch_size, hidden_size]
        decoder_input = torch.randn(batch_size, self.num_nodes * self.output_dim).to(self.device)
        # decoder_input = [batch_size, self.num_nodes * self.output_dim]

        outputs = []
        for i in range(self.output_window):
            decoder_output, decoder_hn, decoder_cn = \
                self.decoder(decoder_input, decoder_hn, decoder_cn)
            # decoder_output = [batch_size, self.num_nodes * self.output_dim]
            # decoder_hn = [layers * num_directions, batch_size, hidden_size]
            outputs.append(decoder_output.reshape(batch_size, self.num_nodes, self.output_dim))
            # 只有训练的时候才考虑用真值
            if self.training and random.random() < self.teacher_forcing_ratio:
                decoder_input = target[i]
            else:
                decoder_input = decoder_output
        outputs = torch.stack(outputs)
        # outputs = [self.output_window, batch_size, self.num_nodes, self.output_dim]
        return outputs.permute(1, 0, 2, 3)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
