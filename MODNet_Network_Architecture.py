from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F



class modnet_encoder(nn.Module):
    def __init__(self, input_dim=3, patch_nums=400, sym_op='max',patch_num=3):
        super(modnet_encoder, self).__init__()
        self.patch_nums = patch_nums
        self.sym_op = sym_op
        self.input_dim = input_dim
        self.patch_num = patch_num

        self.conv1_1 = nn.Conv1d(self.input_dim, 64, kernel_size=1)
        self.conv1_2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv1_3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv1_4 = nn.Conv1d(256, 512, kernel_size=1)
        self.conv1_5 = nn.Conv1d(512, 512, kernel_size=1)

        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn1_3 = nn.BatchNorm1d(256)
        self.bn1_4 = nn.BatchNorm1d(512)
        self.bn1_5 = nn.BatchNorm1d(512)

        self.conv2_1 = nn.Conv1d(self.input_dim, 64, kernel_size=1)
        self.conv2_2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv2_3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv2_4 = nn.Conv1d(256, 512, kernel_size=1)
        self.conv2_5 = nn.Conv1d(512, 512, kernel_size=1)

        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(256)
        self.bn2_4 = nn.BatchNorm1d(512)
        self.bn2_5 = nn.BatchNorm1d(512)

        self.conv3_1 = nn.Conv1d(self.input_dim, 64, kernel_size=1)
        self.conv3_2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3_3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv3_4 = nn.Conv1d(256, 512, kernel_size=1)
        self.conv3_5 = nn.Conv1d(512, 512, kernel_size=1)

        self.bn3_1 = nn.BatchNorm1d(64)
        self.bn3_2 = nn.BatchNorm1d(128)
        self.bn3_3 = nn.BatchNorm1d(256)
        self.bn3_4 = nn.BatchNorm1d(512)
        self.bn3_5 = nn.BatchNorm1d(512)

        self.activate = nn.ReLU()



    def forward(self, x):


        x_1=x[:,0,:,:]
        x_2=x[:,1,:,:]
        x_3=x[:,2,:,:]
        x_1 = self.activate(self.bn1_1(self.conv1_1(x_1)))
        x_1 = self.activate(self.bn1_2(self.conv1_2(x_1)))
        x_1 = self.activate(self.bn1_3(self.conv1_3(x_1)))
        x_1 = self.activate(self.bn1_4(self.conv1_4(x_1)))
        x_1 = self.activate(self.bn1_5(self.conv1_5(x_1)))

        x_2 = self.activate(self.bn2_1(self.conv2_1(x_2)))
        x_2 = self.activate(self.bn2_2(self.conv2_2(x_2)))
        x_2 = self.activate(self.bn2_3(self.conv2_3(x_2)))
        x_2 = self.activate(self.bn2_4(self.conv2_4(x_2)))
        x_2 = self.activate(self.bn2_5(self.conv2_5(x_2)))

        x_3 = self.activate(self.bn3_1(self.conv3_1(x_3)))
        x_3 = self.activate(self.bn3_2(self.conv3_2(x_3)))
        x_3 = self.activate(self.bn3_3(self.conv3_3(x_3)))
        x_3 = self.activate(self.bn3_4(self.conv3_4(x_3)))
        x_3 = self.activate(self.bn3_5(self.conv3_5(x_3)))

        if self.sym_op == 'sum':
            x_1 = torch.sum(x_1, dim=-1)
            x_2 = torch.sum(x_2, dim=-1)
            x_3 = torch.sum(x_3, dim=-1)

        else:
            x_1, index_1 = torch.max(x_1, dim=-1)
            x_2, index_2 = torch.max(x_2, dim=-1)
            x_3, index_3 = torch.max(x_3, dim=-1)

        x_1 = x_1.unsqueeze(1)
        x_2 = x_2.unsqueeze(1)
        x_3 = x_3.unsqueeze(1)

        output = x_1
        output = torch.cat((x_2, output), dim=-2)
        output = torch.cat((x_3, output), dim=-2)





        return output


class modnet_decoder(nn.Module):
    def __init__(self,patch_num=5):
        super(modnet_decoder, self).__init__()
        self.patch_num = patch_num

        self.fc1_1 = nn.Linear(512, 512)
        self.fc1_2 = nn.Linear(512, 256)
        self.fc1_3 = nn.Linear(256, 3)
        self.fc1_4 = nn.Linear(256, 3)
        self.bn1_1 = nn.BatchNorm1d(512)
        self.bn1_2 = nn.BatchNorm1d(256)
        self.dropout1_1 = nn.Dropout(0.3)
        self.dropout1_2 = nn.Dropout(0.3)

        self.fc2_1 = nn.Linear(512, 512)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc2_3 = nn.Linear(256, 3)
        self.fc2_4= nn.Linear(256, 3)
        self.bn2_1 = nn.BatchNorm1d(512)
        self.bn2_2 = nn.BatchNorm1d(256)
        self.dropout2_1 = nn.Dropout(0.3)
        self.dropout2_2 = nn.Dropout(0.3)


        self.fc3_1 = nn.Linear(512, 512)
        self.fc3_2 = nn.Linear(512, 256)
        self.fc3_3 = nn.Linear(256, 3)
        self.fc3_4 = nn.Linear(256, 3)
        self.bn3_1 = nn.BatchNorm1d(512)
        self.bn3_2 = nn.BatchNorm1d(256)
        self.dropout3_1 = nn.Dropout(0.3)
        self.dropout3_2 = nn.Dropout(0.3)

        self.global_fc1_attention = nn.Linear(1536, 512)
        self.global_bn1 = nn.BatchNorm1d(512)

        self.global_fc2_attention = nn.Linear(512, 128)
        self.global_bn2 = nn.BatchNorm1d(128)
        self.global_fc3_attention = nn.Linear(128, 9)


        self.fc1_attention = nn.Linear(512, 512)
        self.fc2_attention = nn.Linear(512, 512)
        self.fc3_attention = nn.Linear(512, 512)

        self.relu = nn.ReLU()






    def forward(self, x):
        x_1=x[:,0,:]#min
        x_2=x[:,1,:]
        x_3=x[:,2,:]#max


        global_feature= torch.cat((x_1, x_2), dim=-1)
        global_feature= torch.cat((global_feature, x_3), dim=-1)
        global_feature = F.relu(self.global_bn1(self. global_fc1_attention(global_feature)))


        global_feature_1 = torch.sigmoid(self.fc1_attention(global_feature))
        global_feature_2 = torch.sigmoid(self.fc2_attention(global_feature))
        global_feature_3 = torch.sigmoid(self.fc3_attention(global_feature))



        x_1=global_feature_1*x_1
        x_2=global_feature_2*x_2
        x_3=global_feature_3*x_3


        x_1 = F.relu(self.bn1_1(self.fc1_1(x_1)))
        x_1 = self.dropout1_1(x_1)
        x_1 = F.relu(self.bn1_2(self.fc1_2(x_1)))
        x_1_temp = self.dropout1_2(x_1)
        x_1 = torch.tanh(self.fc1_3(x_1_temp))
        x_1_temp = torch.tanh(self.fc1_4(x_1_temp))



        x_2 = F.relu(self.bn2_1(self.fc2_1(x_2)))
        x_2 = self.dropout2_1(x_2)
        x_2 = F.relu(self.bn2_2(self.fc2_2(x_2)))
        x_2_temp = self.dropout2_2(x_2)
        x_2 = torch.tanh(self.fc2_3(x_2_temp))
        x_2_temp = torch.tanh(self.fc2_4(x_2_temp))


        x_3 = F.relu(self.bn3_1(self.fc3_1(x_3)))
        x_3 = self.dropout3_1(x_3)
        x_3 = F.relu(self.bn3_2(self.fc3_2(x_3)))
        x_3_temp = self.dropout3_2(x_3)
        x_3 = torch.tanh(self.fc3_3(x_3_temp))
        x_3_temp = torch.tanh(self.fc3_4(x_3_temp))



        global_feature = F.relu(self.global_bn2(self.global_fc2_attention(global_feature)))
        offset_weight = self.global_fc3_attention(global_feature)
        offset_weight=torch.softmax(offset_weight.view(-1,3,3),2)


        x_total=x_1_temp*offset_weight[:,0,:]+x_2_temp*offset_weight[:,1,:]+x_3_temp*offset_weight[:,2,:]
        loss_weight=offset_weight.sum(1)/3.0
        loss_weight=torch.mean(loss_weight,0)




        return x_1,x_2,x_3,x_total,loss_weight

class modnet(nn.Module):
    def __init__(self, input_dim=3,patch_num=3, patch_point_nums=400, sym_op='max'):
        super(modnet, self).__init__()

        self.patch_point_nums = patch_point_nums
        self.sym_op = sym_op
        self.input_dim_encoder = input_dim
        self.patch_num = patch_num

        self.encoder = modnet_encoder(self.input_dim_encoder, self.patch_point_nums, self.sym_op)
        self.decoder = modnet_decoder(self.patch_num)

    def forward(self, x):
        x = self.encoder(x)


        x_1,x_2,x_3,x_total,loss_weight = self.decoder(x)

        return  x_1,x_2,x_3,x_total,loss_weight#, encoder_feature


if __name__ == '__main__':




    input = torch.randn(64,3,3,400).cuda()

    model = modnet(patch_num=input.shape[1]).cuda()
    print(model)

    x_1,x_2,x_3,x_total,loss_weight = model(input)
    print(x_3.size())
