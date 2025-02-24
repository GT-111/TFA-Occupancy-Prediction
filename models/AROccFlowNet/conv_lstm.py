import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import load_config
from utils.dataset_utils.I24Motion_utils.generate_test_data import SampleModelInput
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = F.elu(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * F.elu(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, config):
        super(ConvLSTM, self).__init__()
        self.num_layers = config.num_layers

        self.hidden_dim = config.hidden_dim
        self.kernel_size = self._extend_for_multilayer(config.kernel_size, self.num_layers)
        self.hidden_dim = self._extend_for_multilayer(config.hidden_dim, self.num_layers)
        self.input_dim = config.input_dim

        self.batch_first = config.batch_first
        self.bias = config.bias
        self.return_all_layers = config.return_all_layers

        self.cell_list = nn.ModuleList([
            ConvLSTMCell(
                input_dim=self.input_dim if i == 0 else self.hidden_dim[i - 1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            )
            for i in range(self.num_layers)
        ])

    def forward(self, occupancy_map, flow_map, hidden_state=None):
        input_tensor = torch.cat([occupancy_map, flow_map], dim=-3)
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)  # Convert (T, B, C, H, W) -> (B, T, C, H, W)

        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.cell_list[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            return [param] * num_layers
        return param


if __name__ == '__main__':
    config = load_config("configs/AROccFlowNetS.py")
    input_dic = SampleModelInput().generate_sample_input()
    occupancy_map = input_dic['occupancy_map']
    flow_map = input_dic['flow_map']
    input_data = torch.cat([occupancy_map, flow_map], dim=2)
    conv_lstm = ConvLSTM(config.models.convlstm)
    output, last_state = conv_lstm(input_data)

    print("ConvLSTM Output Shape:", output[-1].shape)  # Expected: [2, 5, 128, 256, 256]
    print("ConvLSTM Last Hidden State Shape:", last_state[-1][0].shape)  # Expected: [2, 128, 256, 256]
