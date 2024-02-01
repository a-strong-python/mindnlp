# 写一个transformers/models/regnet/regnet_test.py文件，用于测试regnet模型的正确性
import numpy as np
import torch
import mindspore
from mindspore import nn
from mindspore import Tensor

# 分别导入torch和mindspore版本的regnet模型
from transformers.models.regnet import modeling_regnet as pt
import modeling_regnet as m

dtype_list = [(mindspore.float32, torch.float32)]


class Regnet_test():
    def __init__(self):
        print("<===========Regnet_test===========>")

    def test_RegNetConvLayer(self, ms_dtype, pt_dtype):
        print("<===========test_RegNetConvLayer===========>")
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        stride = 1
        groups = 1
        activation = "relu"
        batch_size = 2
        height = 32
        width = 32
        hidden_size = 3 

        input_shape = (batch_size, in_channels, height, width)
        input_data = torch.randn(input_shape)

         # init model
        ms_model = m.RegNetConvLayer(in_channels, out_channels, kernel_size, stride, groups, activation)
        pt_model = pt.RegNetConvLayer(in_channels, out_channels, kernel_size, stride, groups, activation)

        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))

        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()

        # prepare data
        x = np.random.randn(hidden_size)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)

        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)

        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 1e-5, 1e-5)

        print("<===========test_RegNetConvLayer_end===========>")





   
if __name__ == "__main__":
    t = Regnet_test()
    t.test_RegNetConvLayer(*dtype_list[0])
    # t.test_RegNetEmbeddings(*dtype_list[0])
    # t.test_RegNetShortCut(*dtype_list[0])
    # t.test_RegNetSELayer(*dtype_list[0])
    # t.test_RegNetXLayer(*dtype_list[0])
    # t.test_RegNetYLayer(*dtype_list[0])
    # t.test_RegNetStage(*dtype_list[0])
    # t.test_RegNetEncoder(*dtype_list[0])
    # t.test_RegNetPreTrainedModel(*dtype_list[0])
    # t.test_RegNetModel(*dtype_list[0])
    # t.test_RegNetForImageClassification(*dtype_list[0])