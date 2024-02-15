from SAC.RL_Framework_Mujoco import Muscle_Env
from simulate import Simulate
import config

def main():

    """ Train an agent to control a musculoskeletal system using DRL and RNNs
        ---------------------------------------------------------------------
    * Simulate object runs training
    * Specify whether testing or not in config file, along with the preferred file to save testing statistics
    * To ensure model is saved properly, specify your preferred directories for storing state_dict
    * A list of possible commands with their functions is given in the config.py file as well as the simulate.py file
    * Specify kinematics by choosing kinematics_path in config file, monkey data is currently provided by default
    """

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    ### TRAINING OBJECT ###
    trainer = Simulate(
        Muscle_Env,
        args.model,
        args.gamma,
        args.tau,
        args.lr,
        args.alpha,
        args.automatic_entropy_tuning,
        args.seed,
        args.policy_batch_size,
        args.hidden_size,
        args.policy_replay_size,
        args.multi_policy_loss,
        args.batch_iters,
        args.cuda,
        args.visualize,
        args.root_dir,
        args.checkpoint_file,
        args.checkpoint_folder,
        args.total_episodes,
        args.save_iter,
        args.muscle_path,
        args.muscle_params_path,
        args.kinematics_path,
        args.condition_selection_strategy
    )

    ### TRAIN OR TEST ###
    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.test(args.test_data_filename)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
