import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
device = torch.device("cpu")

#cell of lstm

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state=None):
        
        if cur_state is None:
            size = (input_tensor.size()[2], input_tensor.size()[3])
            batch_size = input_tensor.size()[0]
            h_cur, c_cur = self.init_hidden(batch_size, size)
        else:
            h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



EPS = 1e-7

def inv_sigmoid(x):
    return -torch.log((1-x)/(x+EPS) +EPS) 

def flow_of_batch(batch_x, batch_y, prev):
  #input shape: [batch, img h, img w]
  flows = None
  ma = torch.max(torch.cat((batch_x, batch_y)))
  mi = torch.min(torch.cat((batch_x, batch_y)))
  batch_x =(batch_x - mi) / (ma - mi) * 255 #[batch, img h, img w]
  batch_y =(batch_y - mi) / (ma - mi) * 255
  batch_x = batch_x.squeeze(dim = 1)
  batch_y = batch_y.squeeze(dim = 1)
  new = [0 for _ in range(len(batch_x))]
  for i in range(len(batch_x)):
    x1 = batch_x[i]
    x2 = batch_y[i]
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    new[i] = cv.calcOpticalFlowFarneback(prev=x1,next=x2, flow=prev[i], pyr_scale=0.5, levels=3, winsize=8, iterations=10, poly_n=5, poly_sigma=1.2, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = torch.Tensor(new[i]).to(device)
    if flows is None:
      flows = flow.unsqueeze(dim=0)
    else:
      flows = torch.cat((flows, flow.unsqueeze(dim=0)), dim=0)
  return torch.reshape(flows, (len(batch_x), 2, 128, 128)), new

class ConvLSTMModel(nn.Module):
    def __init__(self, hid_chan, in_chan=1, num_layers=2, use_last = True):
        super(ConvLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.convlstm_layers = nn.ModuleList()
        chans = [in_chan + 2] + [int(hid_chan//2)] + [hid_chan] * num_layers
        for i in range(num_layers):
            cell = ConvLSTMCell(input_dim=chans[i],
                                hidden_dim=chans[i+1],
                                kernel_size=(3, 3),
                                bias=True)
            self.convlstm_layers.append(cell)
        self.input_CNN = nn.Conv2d(in_channels=in_chan + 2,
                                     out_channels=in_chan + 2,
                                     kernel_size=(3, 3),
                                     padding=(1, 1))

        self.decoder_CNN = nn.Conv2d(in_channels=hid_chan + 2,
                                     out_channels=in_chan,
                                     kernel_size=(3, 3),
                                     padding=(1, 1))     
        self.save_fp = "drive/MyDrive/Climate-Hack-2022/Yash/seq2seq/OF+DropoutOnlyLSTM+WD+deep+newOF"
        if use_last:
          self.load_params()
        self = self.to(device)


    def forward(self, inp_imgs, teacher_force=1, teacher_force_start=12):

        """
        Parameters
        ----------
        input_tensor:
            inp_imgs: [batch, num frames, img h, img w]
        output tensor:
            y: [batch, num frames, img h, img w]
        """

        # find size of different input dimensions
        #inp_imgs = inp_imgs.squeeze(dim=0)
        batch_size, frame_cnt, h, w = inp_imgs.size()
        self.frame_cnt = frame_cnt
        inp_imgs = inp_imgs.unsqueeze(dim=2) #[batch, num frames, in chan, img h, img w]

        #setup for optical flow
        flows = torch.zeros((batch_size, 2, h, w)).to(device)
        weight = 1

        outputs = []
        prev = [None for _ in range(batch_size)]
        cur_state = [None for _ in range(self.num_layers)] # [(h0, c0), (h1, c1)...]
        for t in range(self.frame_cnt):
            torch.cuda.empty_cache()
            # generate a mask

            inp = inp_imgs[:, t, :, :, :] # [batch, input channels, img h, img w] # why are we predicitng off a single frame?
            if len(outputs) >= teacher_force_start: #remove eventually #should this be len outputs > 12?
                use_pred = np.random.random([batch_size, 1, 1, 1]) >= teacher_force
                use_pred = torch.Tensor(use_pred)
                use_pred = use_pred.to(device)
                pred_frame = pred[:, -1].unsqueeze(dim=1)
                inp = use_pred * pred_frame + (1 - use_pred) * inp
            inp_img = inp

            #update weighted flow
            if len(outputs) <= 1:
              x1 = inp_imgs[:, 0] # first frame of input #[batch, input channels, img h, img w]
              x2 = inp_imgs[:, 1] # second frame of input
            else:
              x1 = outputs[-1]
              x2 = outputs[-2] #most recent two frames
            flow, prev = flow_of_batch(x1,x2,prev)
            flows = (flows/3 + flow) / weight
            weight += (1/3)**t

            inp = torch.cat((inp, flows), dim = 1)
            inp = self.input_CNN(inp)

            
            for i in range(self.num_layers):
                cur_state[i] = self.convlstm_layers[i](input_tensor=inp, cur_state=cur_state[i])  # we could concat to provide skip conn here
                #add dropout
                inp = nn.Dropout(0.2)(cur_state[i][0])
            final_hid = inp # inp dim: [batch, hid chan, img h, img w]

            encoder_output = torch.cat((final_hid, flows), dim = 1)
            pred = self.decoder_CNN(encoder_output) # [batch, in chan, img h, img w] #change this to take the right dims
            pred = torch.nn.Sigmoid()(inv_sigmoid(inp_img) + pred)
            outputs.append(pred)

        return torch.stack(outputs, dim=1)[:, :, 0, :, :] # [batch, num_frames, img h, img w]

    def ultimate_pred(self, inp_imgs, pred_len=24):
        """
        Parameters
        ----------
        input_tensor:
            inp_imgs: [batch, 12, img h, img w]
        output tensor:
            y: [batch, 24, img h, img w]
        This is the funciton used to make the 12 to 24 final predictions
        """
        batch, inp_len,h,w = inp_imgs.shape
        z = torch.zeros([batch, pred_len - 1, h, w]).to(device)
        aug_inp_img = torch.cat((inp_imgs, z), dim = 1)
        outputs = self(aug_inp_img, teacher_force = 0, teacher_force_start =inp_len)[:, -pred_len:]
        #for i in range(pred_len):
        #    print(i)
        #    out = self(inp_imgs)[:, -1] #[batch, img h, img w]
        #    pred_frame = out.unsqueeze(dim=1) #[batch, 1, img h, img w]
        #    inp_imgs = torch.cat((inp_imgs,pred_frame), dim = 1) #[batch, 12 + i, img h, img w]
        #    outputs.append(out)
        #return torch.stack(outputs, axis=1)
        return outputs
    def load_params(self):
      self.load_state_dict(torch.load(self.save_fp, map_location=device))

    def save_params(self):
      torch.save(self.state_dict(), self.save_fp)