from SAC.RL_Framework_Mujoco import Muscle_Env
from simulate import Simulate
import config

def main():

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    ### Training Object ###
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
        args.muscle_params_path
    )

    ### TRAIN OR TEST ###
    if args.test == False:
        trainer.train()
    else:
        trainer.test(args.test_data_filename)

if __name__ == '__main__':
    main()
