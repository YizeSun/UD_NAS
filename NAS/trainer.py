import os
import torch
import numpy as np
from torch import nn, tensor
import torch.optim as topt
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import json
from circuit_ud import Circuit_manager
from itertools import combinations

TARGET = torch.Tensor([[1., 1., 1.],[1., 1., 1.],[1., 1., 1.],
                       [1., 1., 1.],[1., 1., 1.],[1., 1., 1.],
                       [1., 1., 1.],[1., 1., 1.],])
#TARGET = torch.Tensor([1, 0, 1, 0, 1, 0, 0, 1]) # TODO original for ccnot
TARGET = torch.Tensor([1, 0, 0, 1, 0, 1, 0, 1])
LOSS_FACTOR = 1


class DQAS4RL:
    def __init__(self,
                 qdqn,
                 qdqn_target,
                 gamma,
                 lr,
                 lr_struc,
                 batch_size,
                 greedy,
                 greedy_decay,
                 greedy_min,
                 update_model,
                 update_targ_model,
                 memory_size,
                 max_steps,
                 seed,
                 cm:Circuit_manager,
                 prob_max = 0.5,
                 lr_in=None,
                 lr_out=None,
                 loss_func='MSE',
                 opt='Adam',
                 opt_struc='Adam',
                 device='auto',
                 logging=False,
                 verbose=False,
                 sub_batch_size=1,
                 early_stop=195,
                 structure_batch=10,
                 debug=False,
                 exp_name='default_name',
                 agent_task="default",
                 agent_name="agent0",
                 total_epochs=5000,
                 struc_learning=True,
                 struc_early_stop=0,
                 p_search=False,
                 p_search_lowerbound=2,
                 p_search_period=0,
                 min_loss=0.1):

        self.qdqn = qdqn
        self.qdqn_target = qdqn_target
        self.cm = cm
        
        self.gamma = gamma
        self.lr = lr
        self.lr_struc = lr_struc
        self.lr_in = lr_in
        self.lr_out = lr_out

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.seed = seed

        self.greedy = greedy
        self.greedy_decay = greedy_decay
        self.greedy_min = greedy_min
        
        self.update_model = update_model
        self.update_targ_model = update_targ_model

        self.loss_func_name = loss_func
        # self.loss_func_name_struc = loss_func_struc
        self.opt_name = opt
        self.opt_name_struc = opt_struc
        self.prob_max = prob_max
        self.avcost = 0
        self.early_stop = early_stop
        self.min_loss = min_loss

        self.struc_early_stop = struc_early_stop if struc_early_stop else int(total_epochs / (2 * (int(cm.num_placeholders/cm.learning_step))))
        self.struc_learning = struc_learning
        self.total_epochs = total_epochs

        self.p_search=p_search
        self.p_search_lowerbound=p_search_lowerbound
        self.p_search_period=p_search_period if p_search_period else int(self.struc_early_stop/(2*self.cm.num_ops))

        self.device = 'cpu' if self.cm.noisy else device
        self.logging = logging
        self.verbose = verbose

        self.exp_name = exp_name
        self.agent_task = agent_task
        self.agent_name = agent_name
        self.debug = debug

        self.sub_batch_size = sub_batch_size
        self.structure_batch = structure_batch
        self.best_structs = []
        self.dtype=torch.float32 # TODO think this dtype

        self.config()

    def config(self):
        # device
        if self.device == "auto":
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            # config deactivate tqdm bar
            self.deact_tqdm = True if torch.cuda.is_available() else False
        else:
            self.device = torch.device(self.device)
            # config deactivate tqdm bar
            self.deact_tqdm = True
        
        # set network to device
        self.qdqn = self.qdqn.to(self.device)
        self.qdqn_target = self.qdqn_target.to(self.device)

        # initial struc parameters
        if self.struc_learning:
            v_struc_init = np.zeros([self.cm.current_num_placeholders, self.cm.num_ops], dtype=np.float32)
            self.tmp_ops_candidates = np.arange(self.cm.num_ops, dtype=np.float32)
            self.tmp_ops_candidates = list(np.tile(self.tmp_ops_candidates, (self.cm.current_num_placeholders, 1)))
            # print(f"current num plh: {self.cm.current_num_placeholders}, num ops: {self.cm.num_ops}")
            # print(f"v struc init: {v_struc_init}") # current num plh: 2, num ops: 3
                                                   # v struc init: [[0. 0. 0.]
                                                   #                [0. 0. 0.]]
            self.var_struc=torch.tensor(v_struc_init, requires_grad=True, dtype=self.dtype, device=self.device)
            # print(f"/**//**/*/*/ v struc init: {self.var_struc} /*/*/*/*/*/") # v struc init: tensor([[0., 0., 0.],
                                                                                                    # [0., 0., 0.]], requires_grad=True)
            # set placeholder parameter beta
            # beta_init = np.zeros((self.cm.learning_step,), dtype=np.float32)
            # self.var_beta = torch.tensor(beta_init, requires_grad=True, dtype=self.dtype, device=self.device)

        # set loss functions
        self.loss_func = getattr(nn, self.loss_func_name + 'Loss')()
        # self.loss_func_struc = getattr(nn, self.loss_func_name_struc + 'Loss')()
        torch_opt = getattr(topt, self.opt_name)
        if self.struc_learning:
            torch_opt_struc = getattr(topt, self.opt_name_struc)
            # torch_opt_beta = getattr(topt, self.opt_name_struc)

        params = []
        params.append({'params':self.qdqn.q_layers.parameters()})

        if hasattr(self.qdqn, 'w_input') and self.qdqn.w_input is not None:
            lr_input = self.lr_in if self.lr_in is not None else self.lr
            params.append({'params': self.qdqn.w_input, 'lr': lr_input})

        if hasattr(self.qdqn, 'w_output') and self.qdqn.w_output is not None:
            lr_output = self.lr_out if self.lr_out is not None else self.lr
            params.append({'params': self.qdqn.w_output, 'lr': lr_output})

        # set optimizers
        self.opt = torch_opt(params, lr=self.lr)
        if self.struc_learning:
            self.opt_struc = torch_opt_struc([self.var_struc], lr=self.lr_struc, betas=(0.9, 0.99), weight_decay=0.001)
           #  self.opt_beta = torch_opt_beta([self.var_beta], lr=0.0001, betas=(0.9, 0.99))

        # set lr scheduler
        self.lr_scheduler = lr_scheduler.LinearLR(self.opt, start_factor=0.96, total_iters=110)
        # self.lr_scheduler = lr_scheduler.LinearLR(self.opt, start_factor=0.99, total_iters=1000)
        # 2p u3cu3basic
        # self.lr_scheduler = lr_scheduler.LinearLR(self.opt, start_factor=0.9986, total_iters=3000)
        # self.num_actions = self.env.action_space.n

        if self.logging:
            # exp_name = datetime.now().strftime("DQN-%d_%m_%Y-%H_%M_%S")
            if not os.path.exists('./logs/'):
                os.makedirs('./logs/')
            out_path = self.exp_name + "_" + self.agent_task + "_" + self.agent_name
            self.log_dir = f'./logs/{out_path}/'
            self.reprot_dir = f'./logs/{out_path}/reports/'
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.reprot_dir):
                os.makedirs(self.reprot_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.reset()

    def reset(self):
        self.global_step = 0
        self.epoch_count = 0

    def push_json(self, out, path):
        with open(path, 'w') as f:
            json.dump(out, f,  indent=4)

    def train_structure(self):
        pass

    def make_inputs(self, states, struc):
        return torch.cat((states, struc), 1)
    
    def wrapper(self, input):
        while True:
            yield input
    def preset_byprob(self, prob):
        return np.array([np.random.choice(np.arange(prob.size()[1])
            , p=np.array(prob[i].detach().cpu().clone().numpy())) for i in range(prob.size()[0])])

    def train_model(self, prob):
        if self.debug:
            print(f"** -- start training -- **")
        self.qdqn.train()
        self.opt.zero_grad()
        # TODO usefull?
        # assert self.sub_batch <= self.batch_size
        
        # structure training
        loss_list = []
        deri_struc = []
        # deri_beta = []

        if self.prob_max and self.struc_learning:
            prob = torch.clamp(prob, min=(1-self.prob_max)/self.cm.num_ops, max=self.prob_max)
            prob = torch.transpose(prob.t()/torch.sum(prob, dim=1), 0, 1)
        
        if self.debug and self.struc_learning:
            print(f"** -- prob: {prob}, shape: {prob.shape} -- **")

        grad_params = {}
        if self.struc_learning:
            self.opt_struc.zero_grad()
            sum_sample_strucs = 0
            for idx in range(self.structure_batch):
                # TODO: for multi processing, create a object Transfer, which is used for every single tread, to transfer sample struc from trainer to circuit
                ## -- sample struc --
                sample_struc = self.preset_byprob(prob) # [2, 1, 0] numpy.array
                self.cm.set_current_sampled_struc(sample_struc)
                ## -- dqn --
                # e.g. sample_struc = 
                #                 tensor([[2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0],
                                        # [2, 1, 0]], dtype=torch.int32) with shape(batch size * num_ops)
                # -- predicted --
                states = torch.Tensor([list(map(int, list(format(i, '0'+str(self.cm.num_qubits)+'b')))) for i in range(2**self.cm.num_qubits)])
                predicted = self.qdqn(states)
                # -- target --
                expected_qvalues = TARGET # TODO original for ccnot

                ## -- loss --
                sub_loss = self.loss_func(predicted, expected_qvalues)*LOSS_FACTOR
                if self.debug:
                    print(f"states: {states}]")
                    print(f"predicted: {predicted}]")
                    print(f"expected_qvalues: {expected_qvalues}]")
                    print(f"sub_loss: {sub_loss}]")
                # loss_list.append(sub_loss)
                # if idx >= self.structure_batch/2:
                #     sub_best = min(loss_list)
                #     if sub_best < sub_loss:
                #         sub_loss*=1.01
                # sub loss: 89.35739131702381
                sub_sum = sum(map(self.cm.check_gate, [self.cm.ops[gate][0] for gate in sample_struc] + self.cm.sphc_struc))
                # if self.debug:
                if self.debug:
                    print(f"sample_struc: {sample_struc} -> {[self.cm.ops[gate][0] for gate in sample_struc] + self.cm.sphc_struc}, sub sum: {sub_sum}") # sample_struc: [0 1], sub sum: 3
                sum_sample_strucs += sub_sum
                if sub_sum:
                    sub_loss.backward()
                    ## -- gradient --
                    for name, param in self.qdqn.named_parameters():
                        # print(f"name: {name}, data: {param.grad}")
                        # print(f"grad: {type(param.grad)}")
                        # print(f"grad shape: {self.cm.get_weights_shape()}") # grad shape: (5, 4, 6, 4)
                        if name in ["w_input", "w_output"]: #  structure do not have w_input, w_output, if need uncomment this if conditiion
                            if param.grad is None:
                                gr = torch.zeros(self.qdqn.num_qubits if name == "w_input" else self.qdqn.num_actions)
                            else:
                                gr = param.grad.detach().clone()
                        else:
                            if param.grad is None:
                                gr = torch.from_numpy(np.zeros(self.cm.get_weights_shape(), dtype=np.float32)).type(self.dtype)
                            else:
                                gr = param.grad.detach().clone()
                        summary = grad_params.get(name, [])
                        summary.append(gr)
                        grad_params[name] = summary
                        # print("result: ", grad_params[name])

                loss_list.append(sub_loss)
                with torch.no_grad():
                    # print(f"loss list: {loss_list}") # loss list: [tensor(136.8522, dtype=torch.float64, grad_fn=<MseLossBackward0>)]
                    grad_struc = (-prob).type(self.dtype).to(self.device).index_put(
                                indices=tuple(torch.LongTensor(list(zip(range(self.cm.current_num_placeholders), sample_struc))).t())
                                , values=torch.ones([self.cm.current_num_placeholders], dtype=self.dtype, device=self.device)
                                , accumulate=True).to(self.device)
                    # grad_beta = torch.tensor([torch.exp(torch.sum(self.var_beta))
                    #     *(torch.sum(torch.exp(self.beta))**2-2*torch.sum(torch.exp(self.beta))*beta) for beta in self.var_beta])/(self.var_beta**4)
                    # print(f"grad struc: {grad_struc}")  # tensor([[ 0.6640, -0.3319, -0.3321],
                                                                # [-0.3411,  0.6708, -0.3297]])
                    # deri_struc.append((sub_loss - self.avcost)*grad_struc*torch.prod(prob_beta).to(self.device))
                    deri_struc.append(((sub_loss - self.avcost)*grad_struc).to(self.device))
                    # print(f"deri_struc: {deri_struc}") # deri_struc: [tensor([[-49.8670,  24.9232,  24.9438],
                                                                            # [-49.4827,  24.7241,  24.7586]]), tensor([[ 82.8578, -41.4118, -41.4460],
                                                                            # [-42.5668, -41.0810,  83.6478]]), tensor([[  5.1840, -10.3084,   5.1244],
                                                                            # [  5.2630, -10.3493,   5.0864]]), tensor([[ 49.3190, -24.6493, -24.6696],
                                                                            # [-25.3368,  49.8232, -24.4865]]), tensor([[ 49.3190, -24.6493, -24.6696],
                                                                            # [-25.3368,  49.8232, -24.4865]])]
                    # deri_beta.append((sub_loss - self.avcost)*torch.prod(
                    #         torch.exp(torch.index_select(prob[idx_struc], 0, torch.tensor(sample_struc)))/torch.sum(
                    #             torch.exp(torch.index_select(prob[idx_struc], 0, torch.tensor(sample_struc)))))*grad_beta.to(self.device))
            #print(f"grad params: {grad_params.keys}")
            with torch.no_grad():
                total_loss = torch.mean(torch.stack(loss_list))*LOSS_FACTOR
                self.avcost = torch.mean(torch.stack(loss_list))*LOSS_FACTOR
                if self.debug:
                    print(f"total_loss: {total_loss}")
                    print(f"self.avcost: {self.avcost}")
                    print(f"grad_batch_struc: {grad_batch_struc}") # total_loss: 82.6636263592016
                                                                    # self.avcost: 82.6636263592016
                                                                    # grad_batch_struc: tensor([[-25.1561,  16.2321,   8.9240],
                                                                                    #         [ 20.7435, -25.1252,   4.3818],
                                                                                    #         [-26.1505,  41.1233, -14.9728]], grad_fn=<MeanBackward1>)
            # grad_batch_params = torch.mean(torch.stack(deri_params), dim=0)

            if sum_sample_strucs:
                with torch.no_grad():
                    for name, param in self.qdqn.named_parameters():
                        if name in grad_params.keys():
                            # if self.debug:
                            # print(f"name: {name}")
                            # print(f"tpye: {torch.mean(torch.stack(grad_params[name]), dim=0)}")
                            param.grad=torch.mean(torch.stack(grad_params[name]), dim=0).type(self.dtype).to(self.device) # + penalty_grad_param

                        # if param.grad:
                            #     param.grad = torch.mean(torch.stack(grad_params[name]), dim=0).type(torch.float).to(self.device) # + penalty_grad_param
                            # else:
                            #     param.grad = torch.mean(torch.stack(grad_params[name]), dim=0).to(self.device) # + penalty_grad_param
                            # if self.debug:
                            # print(f"param grad: {param.grad}")
                self.opt.step()
            #print(f"deri struc: {torch.mean(torch.stack(deri_struc), dim=0)}")
                                            # deri struc: tensor([[-67.9110, -21.5343,  89.4454],
                                                                # [-22.7290,  32.3214,  -9.5925],
                                                                # [-22.7290, 101.4352, -78.7062]], grad_fn=<MeanBackward1>)

            # self.var_struc.grad = torch.from_numpy(np.zeros((self.num_placeholders, self.num_ops))).to(self.device)
            # print(f"struc grad: {self.var_struc.grad}")
            # print(f"type: {self.var_struc.grad.type()}")
            with torch.no_grad():
                grad_batch_struc = torch.mean(torch.stack(deri_struc), dim=0).type(self.dtype).to(self.device)
                # grad_batch_beta = torch.mean(torch.stack(deri_beta), dim=0).type(self.dtype).to(self.device)
                if self.debug:
                    print(f"struc grad: {grad_batch_struc}")# struc grad: tensor([[ -99.6193,  -89.2447,  188.8640],
                                                                        # [ -86.9815,   72.4404,   14.5410],
                                                                        # [-102.1552,  232.7695, -130.6143]], grad_fn=<MeanBackward1>)
               #  print(f"struc beta grad: {grad_batch_beta}")
                self.var_struc.grad = grad_batch_struc # + penalty_grad_struc
                # self.var_beta.grad = grad_batch_beta

            self.opt_struc.step()
            # self.opt_beta.step()
        else:
            if self.debug:
                print("** -- parameter learning part -- **")
            self.qdqn.train()
            self.opt.zero_grad()

            # -- predicted --
            states = torch.Tensor([list(map(int, list(format(i, '0'+str(self.cm.num_qubits)+'b')))) for i in range(2**self.cm.num_qubits)])
            predicted = self.qdqn(states)
            # -- target --
            expected_qvalues = TARGET

            ## -- loss --
            if self.debug:
                print(f"states: {states}]")
                print(f"predicted: {predicted}]")
                print(f"expected_qvalues: {expected_qvalues}]")
                print(f"sub_loss: {sub_loss}]")
            ## -- loss --
            total_loss = self.loss_func(predicted, expected_qvalues)
            total_loss.backward()
            # for name, param in self.qdqn.named_parameters():
            #     print(f"name: {name}, data: {param.data}, grad: {param.grad}")
            self.opt.step()

        return total_loss.item()

    def update_target_model(self):
        # for name, param in self.qdqn_target.named_parameters():
        #     print(name, param.data)
        self.qdqn_target.load_state_dict(self.qdqn.state_dict())
        # for name, param in self.qdqn_target.named_parameters():
        #     print(name, param.data)
    def update_prob(self):
        with torch.no_grad():
            #prob_beta = torch.exp(self.var_beta)/torch.sum(torch.exp(self.var_beta))
            prob = torch.zeros(self.var_struc.shape)
            for i in range(self.cm.current_num_placeholders):
                row_sum = torch.sum(torch.exp(self.var_struc[i][self.tmp_ops_candidates[i]]))
                for j, v in enumerate(self.var_struc[i]):
                    if j in self.tmp_ops_candidates[i]:
                        prob[i,j] = torch.exp(v)/row_sum
        return prob

    def epoch_train(self, epoch):
        # TODO: for frozenlake, for other experiment may add step as argument for update_greedy()
        epoch_reward = 0
        epoch_loss = []
        epoch_greedy = 0

        if self.struc_learning:
            with torch.no_grad():
                # prob_beta = torch.exp(self.var_beta)/torch.sum(torch.exp(self.var_beta))
                # 
                # prob_beta: {prob_beta}')
                prob = self.update_prob()
                if self.debug:
                    print(f'****prob****: {prob}, tmp_ops_complement:, {self.tmp_ops_candidates}')
                # progressive leanring
                if self.p_search and epoch % () == 0 and epoch > 0: # for test self.struc_early_stop = 12 self.cm.num_ops = 3
                    for i, row in enumerate(prob):
                        min_ops_idx = -99999
                        v_min = 1
                        #print(f'row: {row}')
                        for j, v in enumerate(row):
                            #print(f'j: {j}, v:{v}, v_min: {v_min}')
                            if v != 0:
                                if v_min > v:
                                    min_ops_idx = j
                                    v_min = v
                            #print(f'j: {j}, v:{v}, v_min: {v_min}, min_ops_idx:{min_ops_idx}')
                            #print(f'this, {self.tmp_ops_candidates}, {self.tmp_ops_candidates[i]}, {len(self.tmp_ops_candidates[i])}')
                        if len(self.tmp_ops_candidates[i]) > self.p_search_lowerbound:
                            #print('before ', self.tmp_ops_candidates[i])
                            self.tmp_ops_candidates[i] = self.tmp_ops_candidates[i][self.tmp_ops_candidates[i] != min_ops_idx]
                            #print('after ', self.tmp_ops_candidates[i])

                # print(f"/**//**/*/*/ prob: {prob} /*/*/*/*/*/") # prob: tensor([[0.3333, 0.3333, 0.3333],
                                                                        # [0.3333, 0.3333, 0.3333]])
        else:
            prob=[]

        # train in batch
        if epoch % self.update_model == 0:
            #loss = self.train_model(prob, prob_beta)
            loss = self.train_model(prob)
            epoch_loss.append(loss)

        self.epoch_count += 1
        self.lr_scheduler.step()
        epoch_avg_loss = np.mean(epoch_loss)
        self.push_json([*self.cm.get_current_layer_struc()], self.log_dir + 'current_struc.json')

        
        # update train
        # p darts
        # update learning place and state
        if epoch > self.struc_early_stop and self.struc_learning:
            # print(f"prob updating : {prob}")
            # learning place and state updated
            prob = self.update_prob()
            layer_learning_finished = self.cm.update_learning_places(prob=prob)
            if layer_learning_finished:
                self.struc_learning = self.cm.learning_state
                if not self.struc_learning: # learning finish
                    return {
                        'steps': epoch,
                        'avg_loss': epoch_avg_loss,
                        'reward': epoch_reward,
                        'greedy': epoch_greedy,
                        'prob': prob.detach().cpu().clone().numpy() # try: prob without all others?
                    }
                else:
                    pass #TODO: next layer or do this in update-learning_places(cm.current_num_placeholders, currrent_learning_places...)
            torch_opt_struc = getattr(topt, self.opt_name_struc)
            v_struc_update_init = np.zeros([self.cm.current_num_placeholders, self.cm.num_ops], dtype=np.float32)
            # print(f"/**//**/*/*/ v struc init: {v_struc_update_init} /*/*/*/*/*/") # [[0. 0. 0.]
                                                                                    #  [0. 0. 0.]
                                                                                    #  [0. 0. 0.]]
            self.var_struc = torch.tensor(v_struc_update_init, requires_grad=True, dtype=self.dtype, device=self.device)
            self.opt_struc = torch_opt_struc([self.var_struc], lr=self.lr_struc)
            self.tmp_ops_candidates = np.arange(self.cm.num_ops, dtype=np.float32)
            self.tmp_ops_candidates = list(np.tile(self.tmp_ops_candidates, (self.cm.current_num_placeholders, 1)))
            self.struc_early_stop += self.struc_early_stop
                # set struc learning false
            # TODO rolling reward for train process
        if self.struc_learning:
            return {
                'steps': epoch,
                'avg_loss': epoch_avg_loss,
                'reward': epoch_reward,
                'greedy': epoch_greedy,
                'prob': prob.detach().cpu().clone().numpy() # try: prob without all others?
            }
        else:
            return {
                'steps': epoch,
                'avg_loss': epoch_avg_loss,
                'reward': epoch_reward,
                'greedy': epoch_greedy,
            }

    def plt_records(self, name, records):
        path = self.reprot_dir + name
        plt.scatter(list(range(len(records))), records, s=0.2)
        plt.title(name)
        plt.ylabel(name)
        plt.xlabel('epochs')
        plt.savefig(path)
        plt.close()

    def save_records(self, name, records):
        path = self.reprot_dir + name
        torch.save(torch.tensor(records), path + '.pt')
        self.plt_records(name, records)
 
    def learn(self,
              num_eval_epochs=10,
              log_train_freq=-1,
              log_eval_freq=-1,
              log_ckp_freq=-1,
              log_records_freq=-1):

        postfix_stats = {}
        records_reward = []
        records_steps = []
        records_avg_loss  = []
        records_greedy = []
        records_probs = []
        avg_rewards = []
        norm_rewards = []
        test_rewards = []

        # for name, param in self.qdqn_target.named_parameters():
        #     print(name, param.data)
        self.push_json(self.cm.ops, self.log_dir + 'operation_pool')

        with tqdm(range(self.total_epochs), desc="DQN",
                  unit="epoch", disable=self.deact_tqdm) as tepochs:
            for t in tepochs:
                if self.debug:
                    print(f'** -- epoch: {t} -- **')

                # train dqn
                train_report = self.epoch_train(t)
                
                postfix_stats['train/epoch_loss'] = train_report['avg_loss']
                postfix_stats['train/epochs'] = train_report['steps']
                postfix_stats['struc-learning'] = self.struc_learning

                records_reward.append(train_report['reward'])
                records_steps.append(train_report['steps'])
                records_avg_loss.append(train_report['avg_loss'])
                records_greedy.append(train_report['greedy'])
                if self.struc_learning:
                    records_probs.append(train_report['prob'].tolist())
                avg_rewards.append(np.mean(records_reward[-100:]))
                norm_rewards.append((records_reward[-1] - np.mean(records_reward[:]))/np.std(records_reward[:]))

                if self.logging and (t % log_train_freq == 0):
                    for key, item in train_report.items():
                        if key != "prob":
                            self.writer.add_scalar('train/' + key, item, t)

                # if self.logging and (t % log_eval_freq == 0):
                #     for key, item in test_report.items():
                #         self.writer.add_scalar('test/' + key, item, t)

                if self.logging and (t % log_ckp_freq == 0):
                    torch.save(self.qdqn.state_dict()
                            , self.log_dir + 'epoch_{}.pt'.format(t))
                    if self.struc_learning:
                        torch.save(self.var_struc
                                , self.log_dir + 'epoch_struc_{}.pt'.format(t))

                # update progress bar
                tepochs.set_postfix(postfix_stats)

                # records during training
                if t % log_records_freq == 0:
                    records = {'reward':records_reward
                            ,'avg_reward': avg_rewards
                            ,'steps': records_steps
                            ,'avg_loss': records_avg_loss
                            ,'greedy':records_greedy
                            , 'norm_reward': norm_rewards
                            , 'test_rewards': test_rewards
                            }
                    # print(f"save sturc1: {[*self.cm.get_learned_layer_struc()]}")
                    if self.struc_learning:
                        self.push_json(records_probs[-10:], self.log_dir + 'last_10_strucs_probs.json')
                        self.push_json([*self.cm.get_learned_layer_struc()], self.log_dir + 'learned_struc.json')
                    for key, value in records.items():
                        self.save_records(key, value)

                if records_avg_loss[-1] <= self.min_loss:
                    output = f"problem solved at epoch {t}, with loss {records_avg_loss[-1]}"
                    if self.struc_learning:
                        output += "learning state {self.struc_learning}"
                    self.push_json(output, self.log_dir + 'problem_solved.json')
                    print(output)
                    break

            if self.logging and (log_ckp_freq > 0):
                torch.save(self.qdqn.state_dict()
                        , self.log_dir + 'episode_final.pt')
                if self.struc_learning:
                    torch.save(self.var_struc
                            , self.log_dir + 'episode_struc_final.pt')

            # final records
            records = {'reward':records_reward
                        ,'avg_reward': avg_rewards
                        ,'steps': records_steps
                        ,'avg_loss': records_avg_loss
                        ,'greedy':records_greedy
                        , 'norm_reward': norm_rewards
                        , 'test_rewards': test_rewards
                        }
            self.push_json([*self.cm.get_learned_layer_struc()], self.log_dir + 'learned_struc.json')
            # print(f"save sturc2: {[*self.cm.get_learned_layer_struc()]}")
            if self.struc_learning:
                self.push_json(records_probs[-10:], self.log_dir + 'last_10_strucs_probs.json')
            for key, value in records.items():
                self.save_records(key, value)
            # for name, param in self.qdqn_target.named_parameters():
            #     print(name, param.data)