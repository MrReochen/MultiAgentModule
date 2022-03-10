import torch
import torch.nn as nn
from ..networks.basic.util import check
from ..networks.basic.predict import PredictNet, PredictLayer, OutLayer
from ..utils.util import get_shape_from_obs_space

class OneHot:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

class Predict(nn.Module):
    def __init__(self, args, obs_space, action_space, cent_obs_space, device=torch.device("cpu")):
        super(Predict, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)


        self.net = PredictNet(args, obs_shape, action_space, cent_obs_shape, use_projector=True)

        self.onehot = OneHot(action_space.n)

        self.to(device)

    def forward(self, obs, actions):
        obs = check(obs).to(**self.tpdv)
        actions = self.onehot.transform(check(actions)).to(**self.tpdv)
        x = torch.cat((obs, actions), dim=-1)
        out = self.net(x)
        return out

class Projector(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Projector, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if args.predict_dim:
            self.mlp = PredictLayer(cent_obs_shape[0], args.predict_dim,
                                1, self._use_orthogonal, False)
            # self.out = OutLayer(cent_obs_shape[0] // 4, self.hidden_size, self._use_orthogonal, False)


        self.to(device)

    def forward(self, cent_obs):
        x = check(cent_obs).to(**self.tpdv)
        if self.args.predict_dim:
            out = self.mlp(x)
            # out = self.out(out)
        else:
            out = x
        return out