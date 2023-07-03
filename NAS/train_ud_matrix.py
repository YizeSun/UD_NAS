import argparse
import timeit
import yaml
import json
from circuit_ud_matrix import QDQN, Circuit_manager
from trainer_matrix import DQAS4RL
from multiprocessing import Pool


parser = argparse.ArgumentParser()

parser.add_argument("--num_layers", default=1, type=int)# *
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lr", default=0.01, type=float) # 0.01 for test of reducing lr
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--greedy", default=1., type=float)
parser.add_argument("--greedy_decay", default=0.99, type=int)
parser.add_argument("--greedy_min", default=0.01, type=float)
parser.add_argument("--update_model", default=1, type=int) # ensure that model updated for every epoch
parser.add_argument("--update_targ_model", default=50, type=int)
parser.add_argument("--memory_size", default=10000, type=int)
parser.add_argument("--loss_func", default='MSE', type=str)
parser.add_argument("--opt", default='Adam', type=str)
parser.add_argument("--epochs_train", default=1000, type=int)# *
parser.add_argument("--epochs_test", default=5, type=int)
parser.add_argument("--device", default='auto', type=str)
parser.add_argument("--early_stop", default=195, type=int)

parser.add_argument("--w_input", default=False, type=bool)
parser.add_argument("--w_output", default=False, type=bool)
parser.add_argument("--lr_input", default=0.001, type=float)
parser.add_argument("--lr_output", default=0.1, type=float)

parser.add_argument("--lr_struc", default=0.01, type=float) # 0.01
parser.add_argument("--max_steps", default=200, type=int) #TODO: fix for fl
parser.add_argument("--seed", default=1234, type=int)
parser.add_argument("--num_placeholders", default=10, type=int)# *
parser.add_argument("--opt_struc", default='Adam', type=str)
parser.add_argument("--structure_batch", default=10, type=int)
parser.add_argument("--num_qubits", default=3, type=int)# *
parser.add_argument("--struc_early_stop", default=0, type=int) # *
parser.add_argument("--learning_step", default=5, type=int) # *
parser.add_argument("--p_search", default=False, type=bool) # *
parser.add_argument("--p_search_lowerbound", default=2, type=int) # *
parser.add_argument("--p_search_period", default=0, type=int) # *

parser.add_argument("--data_reuploading", default=True, type=bool)
parser.add_argument("--use_sphc_struc", default=True, type=bool)
parser.add_argument("--barrier", default=False, type=bool)
parser.add_argument("--exp_name", default='cp', type=str)
parser.add_argument("--agent_task", default='default', type=str)
parser.add_argument("--noisy", default=False, type=bool)

parser.add_argument("--logging", default=True, type=bool)
parser.add_argument("--debug", default=False, type=bool)
parser.add_argument("--log_train_freq", default=1, type=int)
parser.add_argument("--log_eval_freq", default=20, type=int)
parser.add_argument("--log_ckp_freq", default=50, type=int)
parser.add_argument("--log_records_freq", default=50, type=int)

args = parser.parse_args()

def train(exp_name, agent_task, agent_name):
    # print(f"exp name: {exp_name}, agent task: {agent_task}, agent name: {agent_name}")
    start = timeit.default_timer()

    # ops = {0:("RZ", [0]), 1:("RZ", [1]), 2:("RZ", [2])
    #     , 3:("CNOT",[0]), 4:("CNOT",[1])
    #     , 5:("CNOTT", [0])
    #     , 6:("H", [2])
    #     , 7:("E", [0,1,2,3])}
    # ops = {0:("T", [0]), 1:("T", [2])
    #     , 2:("Ta", [1]), 3:("Ta", [2])
    #     , 4:("S", [1])
    #     , 5:("CNOT",[0]), 6:("CNOT",[1])
    #     , 7:("CNOTT", [0])
    #     , 8:("H", [2])
    #     , 9:("E", [0,1,2])}
    ops = {0:("U3", [0]), 1:("U3", [1]), 2:("U3", [2])
        , 3:("CU3-single", [0]), 4:("CU3-single", [1])
        , 5:("CU33", [0])
        , 6:("CNOT",[0]), 7:("CNOT",[1])
        , 8:("CNOTT", [0])
        , 9:("H", [2])
        , 10:("E", [0,1,2])}
    # u3cu3cu33
    ops = {0:("U3", [0]), 1:("U3", [1]), 2:("U3", [2])
        , 3:("CU3-single", [0]), 4:("CU3-single", [1])
        , 5:("CU33", [0])
        , 6:("HCNOT", [1]), 7:("CNOTH", [1])
        , 8:("E", [0,1,2])}
    # rzcnot hcnot
    ops = {0:("RZ", [0])
        , 1:("rz-CNOT-rz", [0]), 2:("rz-CNOT-rz", [1])
        , 3:("rz-CNOTT-rz", [0])
        , 4:("HCNOT", [1]), 5:("CNOTH", [1])
        , 6:("E", [0,1,2])}
    # sym rzcnot
    ops = {0:("RZ", [0])
        , 1:("rz-CNOT", [0]), 2:("CNOT-rz", [0])
        , 3:("rz-CNOT", [1]), 4:("CNOT-rz", [1])
        , 5:("CNOTT", [0])
        , 6:("HCNOT", [1]), 7:("CNOTH", [1])
        , 8:("E", [0,1,2])}
    # sym2 3.22
    ops = {0:("RZ", [0]), 1:("RZ", [1]), 2:("RZ", [2])
        , 3:("rz-CNOT", [0]), 4:("CNOT-rz", [0])
        , 5:("rz-CNOT", [1]), 6:("CNOT-rz", [1])
        , 7:("CNOTT", [0])
        , 8:("H", [2])
        , 9:("E", [0,1,2])}
    # sym3 
    ops = {0:("RZ", [0]), 1:("RZ", [1]), 2:("RZ", [2])
        , 3:("CNOT", [0]), 4:("CNOT", [1])
        , 5:("CNOTT", [0])
        , 6:("H", [2])
        , 7:("E", [0,1,2])}
    # simp1
    # ops = {0:("U3", [0]), 1:("U3", [1]), 2:("U3", [2])
    #     , 3:("CU3-single", [0]), 4:("CU3-single", [1])
    #     , 5:("CU33", [0])
    #     , 6:("H", [2])
    #     , 7:("E", [0,1,2])}

    sphc_struc = []
    # sphc_struc = ["CZ"]
    # sphc_struc = ["RY", "RZ", "CNOT"]
    sphc_ranges = [[*range(args.num_qubits)] for _ in range(len(sphc_struc))]

    cm = Circuit_manager(sphc_struc=sphc_struc
                        , sphc_ranges=sphc_ranges
                        , num_qubits=args.num_qubits
                        , num_placeholders=args.num_placeholders
                        , num_layers=args.num_layers
                        , ops=ops
                        , noisy=args.noisy
                        , learning_step=args.learning_step
                        )

    # Define quantum network
    qdqn = QDQN(cm=cm
            , w_input=args.w_input
            , w_output=args.w_output
            , data_reuploading=args.data_reuploading
            , barrier=args.barrier
            , seed=args.seed)

    qdqn_target = QDQN(cm=cm
                    , w_input=args.w_input
                    , w_output=args.w_output
                    , data_reuploading=args.data_reuploading
                    , barrier=args.barrier
                    , seed=args.seed)

    dqas4rl = DQAS4RL(qdqn=qdqn,
                      qdqn_target=qdqn_target,
                      gamma=args.gamma,
                      lr=args.lr,
                      lr_struc=args.lr_struc,
                      batch_size=args.batch_size,
                      greedy=args.greedy,
                      greedy_decay=args.greedy_decay,
                      greedy_min=args.greedy_min,
                      update_model=args.update_model,
                      update_targ_model=args.update_targ_model,
                      memory_size=args.memory_size,
                      max_steps=args.max_steps,
                      seed=args.seed,
                      cm=cm,
                      prob_max = 0,
                      lr_in=args.lr_input,
                      lr_out=args.lr_output,
                      loss_func=args.loss_func,
                      opt=args.opt,
                      opt_struc=args.opt_struc,
                      device=args.device,
                      logging=args.logging,
                      verbose=False,
                      early_stop=args.early_stop,
                      structure_batch=args.structure_batch,
                      debug=args.debug,
                      exp_name=exp_name,
                      agent_task=agent_task,
                      agent_name=agent_name,
                      struc_learning=cm.learning_state,
                      total_epochs=args.epochs_train,
                      p_search=args.p_search,
                      p_search_lowerbound=args.p_search_lowerbound,
                      p_search_period=args.p_search_period,
                      struc_early_stop=args.struc_early_stop)

    if args.logging:
        with open(dqas4rl.log_dir + 'config.yaml', 'w') as f:
            yaml.safe_dump(args.__dict__, f, indent=2)

    dqas4rl.learn(num_eval_epochs=args.epochs_test,
                  log_train_freq=args.log_train_freq,
                  log_eval_freq=args.log_eval_freq,
                  log_ckp_freq=args.log_ckp_freq,
                  log_records_freq=args.log_records_freq)

    stop = timeit.default_timer()
    with open(dqas4rl.log_dir + 'total_time.json', 'w') as f:
        json.dump(f'total time cost: {stop-start}', f,  indent=4)
    print(f'total time cost: {stop-start}')

def main():
    exp_name=args.exp_name
    agent_task=args.agent_task
    names = ['a0', 'a1', 'a2', 'a3', 'a4']
    # with Pool() as pool:
    #     pool.starmap(train, [(exp_name, agent_task, a) for a in names])
    train(exp_name, agent_task, '0221-matrix-sym3-ep5000-lstep12-lrst001')

if __name__ == '__main__':
    main()