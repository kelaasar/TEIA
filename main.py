from research.config.base_config import parse_argument
from research.data.data_prepare import prepare_pivot_data, prepare_adv_additional_data, \
    prepare_geia_data, prepare_external_data, load_prepared_dataset, load_augmented_data
from research.utils.toolbox import get_free_gpu, same_seed, create_save_path
from research.utils.eval import eval_generation
from research.model.adv_attack_model import SurrogateModel, LLMAttackModel
from torch.utils.data import DataLoader
import wandb
import json
import torch


def get_best_step(val_result):
    '''Get the best step from validation result'''
    best_step = 0
    best_result = 0
    for step, result in enumerate(val_result):
        if result['embed_similarity'] >= best_result:
            best_step = step
            best_result = result['embed_similarity']
    return best_step


def eval_best_result(config, val_result):
    '''Evaluate emb similarity and LLM evaluation'''
    best_step = get_best_step(val_result)
    result_path = create_save_path(config, best_step)
    print(f"Load result from {result_path}")
    with open(result_path, 'r') as f:
        data = json.load(f)
        gt = data['gt']
        pred = data['pred']

    # Evaluate LLM score only
    eval_result = eval_generation(pred, gt)

    best_result = {}
    best_val_dict = val_result[best_step]
    for k, v in best_val_dict.items():
        best_result[f'best/{k}'] = v
    for k, v in eval_result.items():
        best_result[f'best/{k}'] = v

    print(best_result)
    if not config['testing']:
        wandb.log(best_result)


if __name__ == '__main__':
    # Initialize settings
    same_seed(123)
    args = parse_argument()
    config = vars(args)
    device = get_free_gpu()

    if not config['testing']:
        # Only monitor when not testing
        wandb.init(project=f"{config['project_name']}",
                   name=f"{config['training_size']}_{config['exp_name']}_{config['blackbox_encoder']}", config=config)
    else:
        # Mini batch config for testing
        config['exp_name'] = 'test'
        config['surrogate_epoch'] = 1
        config['num_epochs'] = 5
        config['eval_per_epochs'] = 1
        config['training_size'] = 10
        config['external_size'] = 10

    print(config)
    train_dataset, val_dataset = load_prepared_dataset(config)
    if not config['geia']:
        train_dataset = load_augmented_data(config, train_dataset)
    external_dataset = prepare_external_data(config)
    private_emb_dim = train_dataset[0][1].shape[0]

    print(f"Number of training data: {len(train_dataset)}.")
    print(f"Number of validation data: {len(val_dataset)}.")
    print(f"Number of external data: {len(external_dataset)}.")

    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=int(config['batch_size'] / 2), pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,
                            shuffle=False,
                            batch_size=config['batch_size'], pin_memory=True)
    print('load data done')

    surrogate = SurrogateModel(config, device)
    # surrogate.fit(train_loader)
    print(f"Without pretrained surrogate model.")

    # Prepare pivot data
    pivot_dataset = prepare_pivot_data(train_dataset, surrogate)
    surrogate_emb_dim = surrogate.output_dim

    # Prepare training data
    if not config['geia']:
        # Prepare new train loader that use additional data from surrogate model
        print(f"Using additional data from adv training")
        final_train_loader = prepare_adv_additional_data(
            pivot_dataset, external_dataset, surrogate, config)
    else:
        print(f"Using GEIA data")
        final_train_loader = prepare_geia_data(pivot_dataset, config)
    del surrogate
    torch.cuda.empty_cache()

    llm_attacker = LLMAttackModel(
        config, private_emb_dim, surrogate_emb_dim, device)
    val_result = llm_attacker.fit(final_train_loader, val_loader)
    del llm_attacker

    eval_best_result(config, val_result)
