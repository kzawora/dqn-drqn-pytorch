import torch
from torch import optim

class DQN(torch.nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.0001):
        super(DQN, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, (8, 8), stride=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (4, 4), stride=(2, 2)),
            torch.nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape))
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(fc_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions)
        )
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        y = self.conv_net(x / 255.0)
        return self.fc_net(y.view(y.shape[0], -1))


class DRQN(torch.nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.0001):
        super(DRQN, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (8, 8), stride=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1)),
            torch.nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape))
            x = self.conv_net(dummy)
            s = x.shape
            lstm_size = s[1] * s[2] * s[3]

        self.lstm = torch.nn.LSTM(lstm_size, 512)
        self.classifier = torch.nn.Linear(512, num_actions)

        self.opt = optim.Adadelta(self.parameters(), lr=lr, rho=0.95)  # was Adam optimizer

    def forward(self, x, hidden=None):
        y = self.conv_net(x / 255.0)
        lstm_in = y.view(y.shape[0], 1, -1)
        y, hidden_out = self.lstm(lstm_in, hidden)
        y = self.classifier(y.squeeze(1))
        return y, hidden_out


class DRQN_shallow(torch.nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.0001):
        super(DRQN_shallow, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (8, 8), stride=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
            torch.nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape))
            x = self.conv_net(dummy)
            s = x.shape
            lstm_size = s[1] * s[2] * s[3]

        self.lstm = torch.nn.LSTM(lstm_size, 256)
        self.classifier = torch.nn.Linear(256, num_actions)

        self.opt = optim.RMSprop(self.parameters(), lr=lr, momentum=0.95)  # was rmsprop optimizer

    def forward(self, x, hidden=None):
        y = self.conv_net(x / 255.0)
        lstm_in = y.view(y.shape[0], 1, -1)
        y, hidden_out = self.lstm(lstm_in, hidden)
        y = self.classifier(y.squeeze(1))
        return y, hidden_out

