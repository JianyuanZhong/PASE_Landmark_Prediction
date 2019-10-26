import torch
import torch.nn as nn

class LandmarkLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.overall_loss = nn.L1Loss()
        self.horizontal_loss = nn.L1Loss()
        self.vertical_loss = nn.L1Loss()
        self.lip_loss = nn.L1Loss()

        self.lip_mask = None
        self.horizental_mask = None
        self.vertical_mask = None

    def forward(self, pred, label):

        if self.lip_mask is None:
            self.lip_mask, self.horizental_mask, self.vertical_mask = self.make_masks(pred, label)

        loss =  0.4 * self.horizontal_loss(pred * self.horizental_mask, label * self.horizental_mask) \
            + 0.4 * self.vertical_loss(pred * self.vertical_mask, label * self.vertical_mask) \
                + 0.8 * self.lip_loss(pred * self.lip_mask, label * self.lip_mask) \
                    + 0.4 * self.overall_loss(pred, label)

        return loss

    def make_masks(self, pred, label):

        original_shape = pred.shape

        pred = pred.view(pred.shape[0], pred.shape[1], pred.shape[2] //2, 2)
        label = label.view(label.shape[0], label.shape[1], label.shape[2] // 2, 2)

        lip_mask = torch.zeros(pred.shape).to(pred.get_device())
        horzental_mask = torch.zeros(label.shape).to(pred.get_device())
        vertical_mask = torch.zeros(label.shape).to(pred.get_device())


        lip_mask[:, :, :47, :] = 1
        horzental_mask[:,:,:,0] = 1
        vertical_mask[:,:,:,1] = 1

        lip_mask = lip_mask.view(original_shape)
        horzental_mask = horzental_mask.view(original_shape)
        vertical_mask = vertical_mask.view(original_shape)

        return lip_mask, horzental_mask, vertical_mask

class RMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, label):
        loss = torch.sqrt(self.loss(pred, label))
        return loss


    